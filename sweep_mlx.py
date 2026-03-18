#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    name: str
    env: dict[str, str]


ARCH_LOCAL_PRESET = [
    Candidate("baseline_9u9_d512", {"NUM_LAYERS": "9", "NUM_UNIQUE_LAYERS": "9", "MODEL_DIM": "512"}),
    Candidate("shared_12u6_d576", {"NUM_LAYERS": "12", "NUM_UNIQUE_LAYERS": "6", "MODEL_DIM": "576"}),
    Candidate("shared_14u7_d576", {"NUM_LAYERS": "14", "NUM_UNIQUE_LAYERS": "7", "MODEL_DIM": "576"}),
    Candidate("shared_12u4_d640", {"NUM_LAYERS": "12", "NUM_UNIQUE_LAYERS": "4", "MODEL_DIM": "640"}),
]

QUANT_LOCAL_PRESET = [
    Candidate("q_default", {"NUM_LAYERS": "12", "NUM_UNIQUE_LAYERS": "4", "MODEL_DIM": "640"}),
    Candidate(
        "q_clip_99999",
        {"NUM_LAYERS": "12", "NUM_UNIQUE_LAYERS": "4", "MODEL_DIM": "640", "INT8_CLIP_PERCENTILE": "99.999"},
    ),
    Candidate(
        "q_clip_999995",
        {"NUM_LAYERS": "12", "NUM_UNIQUE_LAYERS": "4", "MODEL_DIM": "640", "INT8_CLIP_PERCENTILE": "99.9995"},
    ),
    Candidate(
        "q_keep_32768",
        {"NUM_LAYERS": "12", "NUM_UNIQUE_LAYERS": "4", "MODEL_DIM": "640", "INT8_KEEP_FLOAT_MAX_NUMEL": "32768"},
    ),
    Candidate(
        "q_keep_32768_clip_999995",
        {
            "NUM_LAYERS": "12",
            "NUM_UNIQUE_LAYERS": "4",
            "MODEL_DIM": "640",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "32768",
            "INT8_CLIP_PERCENTILE": "99.9995",
        },
    ),
]

HYPER_LOCAL_PRESET = [
    Candidate(
        "h_default",
        {
            "NUM_LAYERS": "12",
            "NUM_UNIQUE_LAYERS": "4",
            "MODEL_DIM": "640",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "32768",
            "INT8_CLIP_PERCENTILE": "99.9995",
        },
    ),
    Candidate(
        "h_lr_down",
        {
            "NUM_LAYERS": "12",
            "NUM_UNIQUE_LAYERS": "4",
            "MODEL_DIM": "640",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "32768",
            "INT8_CLIP_PERCENTILE": "99.9995",
            "MATRIX_LR": "0.035",
            "SCALAR_LR": "0.03",
        },
    ),
    Candidate(
        "h_softcap_20_qkg_125",
        {
            "NUM_LAYERS": "12",
            "NUM_UNIQUE_LAYERS": "4",
            "MODEL_DIM": "640",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "32768",
            "INT8_CLIP_PERCENTILE": "99.9995",
            "LOGIT_SOFTCAP": "20.0",
            "QK_GAIN_INIT": "1.25",
        },
    ),
    Candidate(
        "h_softcap_40_qkg_175",
        {
            "NUM_LAYERS": "12",
            "NUM_UNIQUE_LAYERS": "4",
            "MODEL_DIM": "640",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "32768",
            "INT8_CLIP_PERCENTILE": "99.9995",
            "LOGIT_SOFTCAP": "40.0",
            "QK_GAIN_INIT": "1.75",
        },
    ),
    Candidate(
        "h_embedstd_75e4_muon_975",
        {
            "NUM_LAYERS": "12",
            "NUM_UNIQUE_LAYERS": "4",
            "MODEL_DIM": "640",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "32768",
            "INT8_CLIP_PERCENTILE": "99.9995",
            "TIED_EMBED_INIT_STD": "0.0075",
            "MUON_MOMENTUM": "0.975",
        },
    ),
]

PRESETS = {
    "arch_local": ARCH_LOCAL_PRESET,
    "quant_local": QUANT_LOCAL_PRESET,
    "hyper_local": HYPER_LOCAL_PRESET,
}

VAL_BPB_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)")
SERIALIZED_RE = re.compile(r"serialized_model_int8_zlib:(?P<bytes>\d+) bytes")
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<params>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize local MLX sweeps for Parameter Golf.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="arch_local")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--dataset-path", default="./data/datasets/fineweb10B_sp1024_smoke")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--train-batch-tokens", type=int, default=8192)
    parser.add_argument("--val-batch-size", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--train-log-every", type=int, default=25)
    parser.add_argument("--max-wallclock-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N candidates from the preset")
    parser.add_argument("--prefix", default="", help="Optional prefix added to RUN_IDs")
    return parser.parse_args()


def parse_metrics(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text(encoding="utf-8")
    val_match = VAL_BPB_RE.search(text)
    size_match = SERIALIZED_RE.search(text)
    params_match = MODEL_PARAMS_RE.search(text)
    if val_match is None or size_match is None or params_match is None:
        raise ValueError(f"Could not parse expected metrics from {log_path}")
    return {
        "val_loss": float(val_match.group("val_loss")),
        "val_bpb": float(val_match.group("val_bpb")),
        "serialized_model_int8_zlib_bytes": int(size_match.group("bytes")),
        "model_params": int(params_match.group("params")),
    }


def run_candidate(repo_dir: Path, base_env: dict[str, str], run_id: str, candidate: Candidate) -> dict[str, object]:
    env = deepcopy(base_env)
    env.update(candidate.env)
    env["RUN_ID"] = run_id
    log_path = repo_dir / "logs" / f"{run_id}.txt"
    if log_path.exists():
        log_path.unlink()

    cmd = [sys.executable, "train_gpt_mlx.py"]
    print(f"\n=== Running {run_id} ===")
    print(" ".join(f"{k}={v}" for k, v in sorted(candidate.env.items())))
    subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
    metrics = parse_metrics(log_path)
    return {
        "run_id": run_id,
        "candidate": candidate.name,
        "env": candidate.env,
        **metrics,
        "log_path": str(log_path),
    }


def main() -> None:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    sweep_dir = repo_dir / "logs" / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.prefix}_" if args.prefix else ""
    candidates = PRESETS[args.preset]
    if args.limit > 0:
        candidates = candidates[: args.limit]

    base_env = os.environ.copy()
    base_env.update(
        {
            "DATA_PATH": args.dataset_path,
            "TOKENIZER_PATH": args.tokenizer_path,
            "ITERATIONS": str(args.iterations),
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "VAL_BATCH_SIZE": str(args.val_batch_size),
            "VAL_LOSS_EVERY": "0",
            "TRAIN_LOG_EVERY": str(args.train_log_every),
            "WARMUP_STEPS": str(args.warmup_steps),
            "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
        }
    )

    results: list[dict[str, object]] = []
    for candidate in candidates:
        run_id = f"{prefix}{args.preset}_{candidate.name}_{timestamp}"
        try:
            results.append(run_candidate(repo_dir, base_env, run_id, candidate))
        except subprocess.CalledProcessError as exc:
            results.append(
                {
                    "run_id": run_id,
                    "candidate": candidate.name,
                    "env": candidate.env,
                    "error": f"returncode {exc.returncode}",
                }
            )

    json_path = sweep_dir / f"{timestamp}_{args.preset}.json"
    csv_path = sweep_dir / f"{timestamp}_{args.preset}.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fieldnames = [
        "candidate",
        "run_id",
        "val_bpb",
        "val_loss",
        "serialized_model_int8_zlib_bytes",
        "model_params",
        "log_path",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    completed = [row for row in results if "val_bpb" in row]
    completed.sort(key=lambda row: float(row["val_bpb"]))
    print("\n=== Sweep Summary ===")
    for row in completed:
        print(
            f"{row['candidate']}: val_bpb={row['val_bpb']:.8f} "
            f"val_loss={row['val_loss']:.8f} "
            f"int8_zlib={row['serialized_model_int8_zlib_bytes']} "
            f"params={row['model_params']}"
        )
    failed = [row for row in results if "error" in row]
    for row in failed:
        print(f"{row['candidate']}: FAILED {row['error']}")
    print(f"\nSaved summary to {json_path}")
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
