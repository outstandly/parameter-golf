#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
from mlx.utils import tree_unflatten

from train_gpt_mlx import GPT, Hyperparameters, dequantize_state_dict_int8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a saved MLX Parameter Golf checkpoint.")
    parser.add_argument("--model-path", required=True, help="Path to a quantized .int8.ptz checkpoint")
    parser.add_argument("--prompt", required=True, help="Prompt text to continue from")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Number of new tokens to sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature; use 0 for greedy")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k truncation before sampling; 0 disables it")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    return parser.parse_args()


def build_model(cfg: Hyperparameters) -> GPT:
    return GPT(
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        num_unique_layers=cfg.num_unique_layers,
        dim=cfg.model_dim,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        mlp_mult=cfg.mlp_mult,
        logit_chunk_tokens=cfg.logit_chunk_tokens,
        logit_softcap=cfg.logit_softcap,
        rope_base=cfg.rope_base,
        tied_embed_init_std=cfg.tied_embed_init_std,
        qk_gain_init=cfg.qk_gain_init,
    )


def load_quantized_checkpoint(model: GPT, model_path: Path) -> None:
    with model_path.open("rb") as f:
        quant_obj = pickle.loads(zlib.decompress(f.read()))
    flat_state = dequantize_state_dict_int8(quant_obj)
    model.update(tree_unflatten(list(flat_state.items())))


def sample_next_token(logits: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    scores = np.asarray(logits, dtype=np.float32).copy()
    if top_k > 0 and top_k < scores.size:
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        masked = np.full_like(scores, -np.inf)
        masked[top_idx] = scores[top_idx]
        scores = masked
    if temperature <= 0.0:
        return int(np.argmax(scores))
    scores = scores / temperature
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / probs.sum()
    return int(rng.choice(probs.size, p=probs))


def generate(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    token_ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = mx.array([token_ids], dtype=mx.int32)
        hidden = model(x)
        last_hidden = hidden[:, -1, :]
        logits = model.softcap(last_hidden @ model.tok_emb.weight.astype(last_hidden.dtype).T).astype(mx.float32)
        mx.eval(logits)
        next_id = sample_next_token(np.array(logits[0]), temperature=temperature, top_k=top_k, rng=rng)
        token_ids.append(next_id)
    return token_ids


def main() -> None:
    args = parse_args()
    cfg = Hyperparameters()
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    prompt_ids = sp.encode(args.prompt, out_type=int)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens; please provide a non-empty prompt.")

    model = build_model(cfg)
    load_quantized_checkpoint(model, model_path)

    token_ids = generate(
        model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    generated_ids = token_ids[len(prompt_ids):]
    print("Prompt:")
    print(args.prompt)
    print("\nCompletion:")
    print(sp.decode(generated_ids))


if __name__ == "__main__":
    main()
