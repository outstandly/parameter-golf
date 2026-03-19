#!/usr/bin/env python3
from __future__ import annotations

import copy
import os
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import train_gpt as tg


def main() -> None:
    args = tg.Hyperparameters()
    profile_steps = int(os.environ.get("PROFILE_STEPS", 20))
    out_dir = Path(os.environ.get("PROFILE_OUT_DIR", "./logs/profiles")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", 0)
    world_size = 1
    grad_accum_steps = 8
    grad_scale = 1.0 / grad_accum_steps

    base_model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.num_unique_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, tg.CastedLinear):
            module.float()
    tg.restore_low_dim_params_to_fp32(base_model)
    model = torch.compile(base_model, dynamic=False, fullgraph=True)

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in tg.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in tg.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.uses_shared_layer_weights:
        scalar_params.extend([base_model.attn_scales, base_model.mlp_scales, base_model.resid_mixes])
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = tg.Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    train_loader = tg.DistributedTokenLoader(args.train_files, 0, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    print("=== Profile Config ===")
    print(f"torch: {torch.__version__}")
    print(f"profile_steps: {profile_steps}")
    print(f"num_layers: {args.num_layers}")
    print(f"num_kv_heads: {args.num_kv_heads}")
    print(f"model_dim: {args.model_dim}")
    print(f"train_batch_tokens: {args.train_batch_tokens}")
    print(f"train_seq_len: {args.train_seq_len}")
    print(f"warmup_steps: {args.warmup_steps}")
    print()

    # Warm compiled graph and optimizer paths, then restore to true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                print(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = tg.DistributedTokenLoader(args.train_files, 0, world_size, device)

    step_times_ms: list[float] = []
    trace_path = out_dir / f"{args.run_id}_profile_trace.json"

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step_idx in range(profile_steps):
            step_t0 = time.perf_counter()
            zero_grad_all()
            train_loss = 0.0
            for micro_idx in range(grad_accum_steps):
                with record_function("data.next_batch"):
                    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with record_function("train.forward_backward"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        loss = model(x, y)
                    (loss * grad_scale).backward()
                train_loss += float(loss.detach().item())

            with record_function("optimizer.step_all"):
                for opt in optimizers:
                    opt.step()
            zero_grad_all()
            torch.cuda.synchronize()
            step_times_ms.append(1000.0 * (time.perf_counter() - step_t0))
            prof.step()
            print(
                f"profile_step:{step_idx + 1}/{profile_steps} "
                f"train_loss:{train_loss / grad_accum_steps:.4f} "
                f"step_time_ms:{step_times_ms[-1]:.2f}"
            )

    prof.export_chrome_trace(str(trace_path))

    print()
    print("=== Step Timing ===")
    print(f"mean_step_ms: {statistics.mean(step_times_ms):.2f}")
    print(f"median_step_ms: {statistics.median(step_times_ms):.2f}")
    if len(step_times_ms) > 1:
        print(f"stdev_step_ms: {statistics.pstdev(step_times_ms):.2f}")
    print(f"peak_memory_allocated_mib: {torch.cuda.max_memory_allocated() / 1024 / 1024:.0f}")
    print()

    print("=== CUDA Time Table ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
    print()
    print("=== CPU Time Table ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
    print()
    print(f"chrome_trace: {trace_path}")


if __name__ == "__main__":
    main()
