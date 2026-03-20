"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from init_utils import overtone_spectral_init_
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    load_state_dict_path = os.environ.get("LOAD_STATE_DICT_PATH", "")
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    skip_preexport_eval = bool(int(os.environ.get("SKIP_PREEXPORT_EVAL", "0")))
    eval_mode = os.environ.get("EVAL_MODE", "chunked")
    eval_doc_isolated = bool(int(os.environ.get("EVAL_DOC_ISOLATED", "0")))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", os.environ.get("TRAIN_SEQ_LEN", 1024)))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", os.environ.get("NUM_LAYERS", 9)))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.0))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, weight_decay=weight_decay, nesterov=nesterov),
        )
        self._group_offsets: list[list[tuple[int, int]]] = []
        self._flat_buffers: list[Tensor | None] = []
        for group in self.param_groups:
            offsets: list[tuple[int, int]] = []
            curr = 0
            for p in group["params"]:
                next_curr = curr + int(p.numel())
                offsets.append((curr, next_curr))
                curr = next_curr
            self._group_offsets.append(offsets)
            self._flat_buffers.append(None)

    def _ensure_group_layout(self, group_idx: int, params: list[Tensor]) -> tuple[list[tuple[int, int]], int]:
        if group_idx >= len(self._group_offsets):
            self._group_offsets.extend([] for _ in range(group_idx + 1 - len(self._group_offsets)))
            self._flat_buffers.extend([None] * (group_idx + 1 - len(self._flat_buffers)))
        offsets = self._group_offsets[group_idx]
        expected = len(params)
        if len(offsets) != expected or any((end - start) != int(p.numel()) for (start, end), p in zip(offsets, params)):
            offsets = []
            curr = 0
            for p in params:
                next_curr = curr + int(p.numel())
                offsets.append((curr, next_curr))
                curr = next_curr
            self._group_offsets[group_idx] = offsets
            self._flat_buffers[group_idx] = None
        total_params = offsets[-1][1] if offsets else 0
        return offsets, total_params

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group_idx, group in enumerate(self.param_groups):
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            if world_size == 1:
                for p in params:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    if g.dtype != p.dtype:
                        g = g.to(dtype=p.dtype)
                    if weight_decay > 0:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(g, alpha=-lr)
                continue

            offsets, total_params = self._ensure_group_layout(group_idx, params)
            updates_flat = self._flat_buffers[group_idx]
            if updates_flat is None or updates_flat.numel() != total_params or updates_flat.device != params[0].device:
                updates_flat = torch.empty(total_params, device=params[0].device, dtype=torch.bfloat16)
                self._flat_buffers[group_idx] = updates_flat
            updates_flat.zero_()

            for i, p in enumerate(params):
                if i % world_size != rank or p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                start, end = offsets[i]
                updates_flat[start:end].copy_(g.reshape(-1))

            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            for p, (start, end) in zip(params, offsets):
                g = updates_flat[start:end].view_as(p)
                if g.dtype != p.dtype:
                    g = g.to(dtype=p.dtype)
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(g, alpha=-lr)

        return loss


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if tokens.numel() <= 1:
        raise ValueError("Validation split is too short")
    return tokens


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    eval_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    bos_token_id: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens <= 0:
        raise ValueError(
            f"VAL_BATCH_SIZE={args.val_batch_size} must provide positive local tokens for "
            f"WORLD_SIZE={world_size} and GRAD_ACCUM_STEPS={grad_accum_steps}"
        )
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    fast_chunked_eval = (
        args.eval_mode == "chunked"
        and not args.eval_doc_isolated
        and args.eval_seq_len == args.train_seq_len
    )
    model.eval()
    eval_model.eval()
    with torch.inference_mode():
        if fast_chunked_eval:
            if local_batch_tokens < args.train_seq_len:
                raise ValueError(
                    "VAL_BATCH_SIZE must provide at least one sequence per rank; "
                    f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
                    f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
                )
            local_batch_seqs = local_batch_tokens // args.train_seq_len
            total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
            seq_start = (total_seqs * rank) // world_size
            seq_end = (total_seqs * (rank + 1)) // world_size
            usable = total_seqs * args.train_seq_len + 1
            val_tokens = val_tokens[:usable]
            for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
                batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
                raw_start = batch_seq_start * args.train_seq_len
                raw_end = batch_seq_end * args.train_seq_len + 1
                local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
                x = local[:-1].reshape(-1, args.train_seq_len)
                y = local[1:].reshape(-1, args.train_seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y).detach()
                batch_token_count = float(y.numel())
                val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
                val_token_count += batch_token_count
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()
        else:
            if args.eval_mode not in {"chunked", "sliding"}:
                raise ValueError(f"Unsupported EVAL_MODE={args.eval_mode}")
            if args.eval_seq_len <= 0:
                raise ValueError(f"EVAL_SEQ_LEN must be positive, got {args.eval_seq_len}")
            if args.eval_mode == "sliding":
                if args.eval_doc_isolated:
                    raise ValueError("Sliding eval does not support EVAL_DOC_ISOLATED")
                if args.eval_stride <= 0:
                    raise ValueError(f"EVAL_STRIDE must be positive for sliding eval, got {args.eval_stride}")
                total = val_tokens.numel() - 1
                windows: list[tuple[int, int]] = []
                p = 0
                while p + args.eval_seq_len <= total:
                    windows.append((p, 0 if p == 0 else (args.eval_seq_len - args.eval_stride)))
                    p += args.eval_stride
                per_rank = (len(windows) + world_size - 1) // world_size
                my_windows = windows[rank * per_rank : min((rank + 1) * per_rank, len(windows))]
                eval_batch_seqs = 256
                for i in range(0, len(my_windows), eval_batch_seqs):
                    batch = my_windows[i : i + eval_batch_seqs]
                    if not batch:
                        continue
                    x_list = [val_tokens[w : w + args.eval_seq_len] for w, _ in batch]
                    y_list = [val_tokens[w + 1 : w + args.eval_seq_len + 1] for w, _ in batch]
                    while len(x_list) < eval_batch_seqs:
                        x_list.append(x_list[-1])
                        y_list.append(y_list[-1])
                    x_batch = torch.stack(x_list).to(device=device, dtype=torch.int64, non_blocking=True)
                    y_batch = torch.stack(y_list).to(device=device, dtype=torch.int64, non_blocking=True)
                    score_mask = torch.zeros((eval_batch_seqs, args.eval_seq_len), device=device, dtype=torch.bool)
                    for j, (_, score_start) in enumerate(batch):
                        score_mask[j, score_start:] = True
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        logits = eval_model.forward_logits(x_batch)
                    per_token_loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        y_batch.reshape(-1),
                        reduction="none",
                    ).reshape_as(y_batch)
                    val_loss_sum += per_token_loss[score_mask].to(torch.float64).sum()
                    val_token_count += score_mask.sum().to(torch.float64)
                    prev_ids = x_batch[score_mask]
                    tgt_ids = y_batch[score_mask]
                    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                    token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                    val_byte_count += token_bytes.to(torch.float64).sum()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
                model.train()
                return float((val_loss_sum / val_token_count).item()), float((val_loss_sum.item() / math.log(2.0)) / val_byte_count.item())
            chunk_targets = args.eval_seq_len
            batch_items: list[tuple[Tensor, Tensor, int]] = []
            batch_input_tokens = 0
            spans: list[tuple[int, int, int, int]] = []

            if args.eval_doc_isolated:
                if bos_token_id < 0:
                    raise ValueError("SentencePiece tokenizer must define BOS for doc-isolated eval")
                doc_starts = (val_tokens == bos_token_id).nonzero(as_tuple=False).flatten().tolist()
                if not doc_starts:
                    raise ValueError("Could not find any BOS-delimited validation documents")
                if doc_starts[0] != 0:
                    doc_starts.insert(0, 0)
                doc_ends = [*doc_starts[1:], int(val_tokens.numel())]
                doc_ranges = list(zip(doc_starts, doc_ends, strict=True))
                doc_start = (len(doc_ranges) * rank) // world_size
                doc_end = (len(doc_ranges) * (rank + 1)) // world_size
                spans = [(raw_start, raw_end, raw_start + 1, raw_end) for raw_start, raw_end in doc_ranges[doc_start:doc_end]]
            else:
                total_targets = val_tokens.numel() - 1
                spans = [(0, int(val_tokens.numel()), 1 + (total_targets * rank) // world_size, 1 + (total_targets * (rank + 1)) // world_size)]

            def flush_batch() -> None:
                nonlocal batch_items, batch_input_tokens, val_loss_sum, val_token_count, val_byte_count
                if not batch_items:
                    return
                max_len = max(x.numel() for x, _, _ in batch_items)
                x_batch = torch.full((len(batch_items), max_len), 0, device=device, dtype=torch.int64)
                y_batch = torch.full((len(batch_items), max_len), 0, device=device, dtype=torch.int64)
                score_mask = torch.zeros((len(batch_items), max_len), device=device, dtype=torch.bool)
                for i, (x, y, score_start) in enumerate(batch_items):
                    n = x.numel()
                    x_batch[i, :n] = x.to(device=device, dtype=torch.int64, non_blocking=True)
                    y_batch[i, :n] = y.to(device=device, dtype=torch.int64, non_blocking=True)
                    score_mask[i, score_start:n] = True
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = eval_model.forward_logits(x_batch)
                per_token_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y_batch.reshape(-1), reduction="none").reshape_as(y_batch)
                val_loss_sum += per_token_loss[score_mask].to(torch.float64).sum()
                val_token_count += score_mask.sum().to(torch.float64)
                prev_ids = x_batch[score_mask]
                tgt_ids = y_batch[score_mask]
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()
                batch_items = []
                batch_input_tokens = 0

            for span_start, _, target_start, target_limit in spans:
                while target_start < target_limit:
                    target_end = min(target_start + chunk_targets, target_limit)
                    window = val_tokens[target_start - 1 : target_end]
                    x = window[:-1]
                    y = window[1:]
                    if batch_items and batch_input_tokens + int(x.numel()) > local_batch_tokens:
                        flush_batch()
                    batch_items.append((x, y, 0))
                    batch_input_tokens += int(x.numel())
                    target_start = target_end
            flush_batch()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_NAME_PATTERNS",
        "tok_emb.weight",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_PATTERN_CANDIDATES = tuple(
    tuple(pattern for pattern in candidate.split(",") if pattern)
    for candidate in os.environ.get(
        "INT8_KEEP_FLOAT_PATTERN_CANDIDATES",
        (
            "tok_emb.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight,blocks.8.attn.proj.weight;"
            "tok_emb.weight,blocks.8.mlp.fc.weight,blocks.8.mlp.proj.weight;"
            "tok_emb.weight,blocks.8.mlp.fc.weight,blocks.8.attn.proj.weight;"
            "tok_emb.weight,blocks.8.mlp.fc.weight;"
            "tok_emb.weight,blocks.8.mlp.proj.weight;"
            "tok_emb.weight,blocks.8.attn.proj.weight;"
            "tok_emb.weight;"
        ),
    ).split(";")
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("INT8_KEEP_FLOAT_MAX_NUMEL", 131_072))
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_GROUP_SIZE = int(os.environ.get("INT8_GROUP_SIZE", 2))
INT8_CLIP_PERCENTILE = float(os.environ.get("INT8_CLIP_PERCENTILE", 99.99995))
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT8_SUBMISSION_SIZE_LIMIT_BYTES = int(os.environ.get("INT8_SUBMISSION_SIZE_LIMIT_BYTES", 16_000_000))

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(
    name: str,
    t: Tensor,
    passthrough_orig_dtypes: dict[str, str],
    keep_float_fp32_name_patterns: tuple[str, ...],
) -> Tensor:
    if any(pattern in name for pattern in keep_float_fp32_name_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor, dict[str, object] | None]:
    t32 = t.float()
    if t32.ndim == 2:
        if INT8_GROUP_SIZE > 0 and t32.shape[1] > INT8_GROUP_SIZE:
            rows, cols = t32.shape
            group_size = INT8_GROUP_SIZE
            num_groups = (cols + group_size - 1) // group_size
            q = torch.empty_like(t32, dtype=torch.int8)
            scales = torch.empty((rows, num_groups), dtype=INT8_PER_ROW_SCALE_DTYPE)
            for group_idx in range(num_groups):
                start = group_idx * group_size
                end = min(start + group_size, cols)
                chunk = t32[:, start:end]
                clip_abs = torch.quantile(chunk.abs(), INT8_CLIP_Q, dim=1) if chunk.numel() else torch.zeros(rows)
                clipped = torch.maximum(torch.minimum(chunk, clip_abs[:, None]), -clip_abs[:, None])
                scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
                q[:, start:end] = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
                scales[:, group_idx] = scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE)
            return q.contiguous(), scales.contiguous(), {"scheme": "per_row_group", "axis": 1, "group_size": group_size}

        # Matrices get one scale per row by default, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(), {"scheme": "per_row", "axis": 0}

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale, None

def quantize_state_dict_int8(
    state_dict: dict[str, Tensor],
    *,
    keep_float_name_patterns: tuple[str, ...] = INT8_KEEP_FLOAT_NAME_PATTERNS,
    keep_float_fp32_name_patterns: tuple[str, ...] = INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if any(pattern in name for pattern in keep_float_name_patterns) or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes, keep_float_fp32_name_patterns)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s, meta = quantize_float_tensor(t)
        if meta is not None:
            qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        if meta.get("scheme") == "per_row_group":
            s = s.to(dtype=torch.float32)
            group_size = int(meta["group_size"])
            rows, cols = q.shape
            out_t = torch.empty((rows, cols), dtype=torch.float32)
            num_groups = s.shape[1]
            for group_idx in range(num_groups):
                start = group_idx * group_size
                end = min(start + group_size, cols)
                out_t[:, start:end] = q[:, start:end].float() * s[:, group_idx].view(rows, 1)
            out[name] = out_t.to(dtype=dtype).contiguous()
        elif meta.get("scheme") == "per_row" or s.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        use_internal_controls: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.use_internal_controls = use_internal_controls
        if use_internal_controls:
            self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
            self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
            self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        else:
            self.register_parameter("attn_scale", None)
            self.register_parameter("mlp_scale", None)
            self.register_parameter("resid_mix", None)

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        resid_mix: Tensor | None = None,
        attn_scale: Tensor | None = None,
        mlp_scale: Tensor | None = None,
    ) -> Tensor:
        if self.use_internal_controls:
            resid_mix = self.resid_mix
            attn_scale = self.attn_scale
            mlp_scale = self.mlp_scale
        elif resid_mix is None or attn_scale is None or mlp_scale is None:
            raise ValueError("Shared-weight blocks require explicit control tensors")
        mix = resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_unique_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if not 1 <= num_unique_layers <= num_layers:
            raise ValueError(
                f"num_unique_layers must be in [1, num_layers], got {num_unique_layers} for num_layers={num_layers}"
            )
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_unique_layers = num_unique_layers
        self.uses_shared_layer_weights = num_unique_layers < num_layers
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        if self.uses_shared_layer_weights:
            self.attn_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
            self.mlp_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
            resid_init = torch.zeros(num_layers, 2, model_dim, dtype=torch.float32)
            resid_init[:, 0, :] = 1.0
            self.resid_mixes = nn.Parameter(resid_init)
            self.blocks = nn.ModuleList(
                [
                    Block(
                        model_dim,
                        num_heads,
                        num_kv_heads,
                        mlp_mult,
                        rope_base,
                        qk_gain_init,
                        use_internal_controls=False,
                    )
                    for _ in range(self.num_unique_layers)
                ]
            )
        else:
            self.register_parameter("attn_scales", None)
            self.register_parameter("mlp_scales", None)
            self.register_parameter("resid_mixes", None)
            self.blocks = nn.ModuleList(
                [
                    Block(
                        model_dim,
                        num_heads,
                        num_kv_heads,
                        mlp_mult,
                        rope_base,
                        qk_gain_init,
                        use_internal_controls=True,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            if int(os.environ.get("OVERTONE_EMBED_INIT", "0")):
                overtone_spectral_init_(self.tok_emb.weight, self.tied_embed_init_std, float(os.environ.get("OVERTONE_EMBED_POWER", 0.5)))
            else:
                nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
        if int(os.environ.get("PHASE_TRANSITION_RESID_INIT", "0")):
            num_layers = len(self.blocks) if not self.uses_shared_layer_weights else self.num_encoder_layers + self.num_decoder_layers
            for i in range(num_layers):
                phase = torch.sigmoid(torch.tensor(3.0 * (i / max(num_layers - 1, 1) - 0.5), dtype=torch.float32))
                if self.uses_shared_layer_weights:
                    self.resid_mixes.data[i, 0].fill_(phase.item())
                    self.resid_mixes.data[i, 1].fill_(1.0 - phase.item())
                else:
                    self.blocks[i].resid_mix.data[0].fill_(phase.item())
                    self.blocks[i].resid_mix.data[1].fill_(1.0 - phase.item())

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        if self.uses_shared_layer_weights:
            for i in range(self.num_encoder_layers):
                x = self.blocks[i % self.num_unique_layers](
                    x,
                    x0,
                    self.resid_mixes[i],
                    self.attn_scales[i],
                    self.mlp_scales[i],
                )
                skips.append(x)
            for i in range(self.num_decoder_layers):
                layer_idx = self.num_encoder_layers + i
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[layer_idx % self.num_unique_layers](
                    x,
                    x0,
                    self.resid_mixes[layer_idx],
                    self.attn_scales[layer_idx],
                    self.mlp_scales[layer_idx],
                )
        else:
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                layer_idx = self.num_encoder_layers + i
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[layer_idx](x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    bos_token_id = int(sp.bos_id())
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(
        f"eval_config:mode={args.eval_mode} doc_isolated={int(args.eval_doc_isolated)} "
        f"eval_seq_len={args.eval_seq_len} eval_stride={args.eval_stride}"
    )

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
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
    if args.load_state_dict_path:
        state_dict = torch.load(args.load_state_dict_path, map_location="cpu")
        base_model.load_state_dict(state_dict, strict=True)
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.uses_shared_layer_weights:
        scalar_params.extend([base_model.attn_scales, base_model.mlp_scales, base_model.resid_mixes])
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
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

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"layers:{args.num_layers} unique_layers:{args.num_unique_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} adam_wd:{args.adam_weight_decay}"
    )
    log0(
        f"init_flags:overtone={int(os.environ.get('OVERTONE_EMBED_INIT', '0'))} "
        f"phase_transition_resid={int(os.environ.get('PHASE_TRANSITION_RESID_INIT', '0'))}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    if args.load_state_dict_path:
        log0(f"loaded_state_dict:{args.load_state_dict_path}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        skip_validate = last_step and args.skip_preexport_eval
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            if skip_validate:
                log0(
                    f"step:{step}/{args.iterations} preexport_eval_skipped:1 "
                    f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
                )
            else:
                val_loss, val_bpb = eval_val(
                    args,
                    model,
                    base_model,
                    rank,
                    world_size,
                    device,
                    grad_accum_steps,
                    val_tokens,
                    bos_token_id,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                log0(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
                )
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_candidates = INT8_KEEP_FLOAT_PATTERN_CANDIDATES if INT8_KEEP_FLOAT_PATTERN_CANDIDATES else (INT8_KEEP_FLOAT_NAME_PATTERNS,)
    seen_candidates: set[tuple[str, ...]] = set()
    ordered_candidates: list[tuple[str, ...]] = []
    for candidate in [*quant_candidates, INT8_KEEP_FLOAT_NAME_PATTERNS, tuple()]:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        ordered_candidates.append(candidate)

    quant_blob = b""
    quant_stats = None
    quant_raw_bytes = 0
    selected_keep_float_patterns: tuple[str, ...] = tuple()
    if master_process:
        code_bytes = len(code.encode("utf-8"))
        best_candidate = None
        state_dict = base_model.state_dict()
        for candidate in ordered_candidates:
            candidate_obj, candidate_stats = quantize_state_dict_int8(
                state_dict,
                keep_float_name_patterns=candidate,
                keep_float_fp32_name_patterns=INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
            )
            candidate_buf = io.BytesIO()
            torch.save(candidate_obj, candidate_buf)
            candidate_raw = candidate_buf.getvalue()
            candidate_blob = zlib.compress(candidate_raw, level=9)
            candidate_total_bytes = len(candidate_blob) + code_bytes
            candidate_label = ",".join(candidate) if candidate else "<none>"
            log0(
                f"int8_candidate keep_float:{candidate_label} "
                f"artifact_bytes:{len(candidate_blob)} total_bytes:{candidate_total_bytes}"
            )
            if candidate_total_bytes <= INT8_SUBMISSION_SIZE_LIMIT_BYTES:
                best_candidate = (candidate, candidate_obj, candidate_stats, candidate_blob, len(candidate_raw))
                break
        if best_candidate is None:
            raise RuntimeError(
                f"No int8 export candidate fit within {INT8_SUBMISSION_SIZE_LIMIT_BYTES} bytes"
            )
        selected_keep_float_patterns, quant_obj, quant_stats, quant_blob, quant_raw_bytes = best_candidate
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        selected_label = ",".join(selected_keep_float_patterns) if selected_keep_float_patterns else "<none>"
        log0(f"int8_selected keep_float:{selected_label}")
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        base_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        bos_token_id,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
