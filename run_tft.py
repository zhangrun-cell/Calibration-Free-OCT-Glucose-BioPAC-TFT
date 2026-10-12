"""TFT-only entry point for the fixed-length prediction implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def build_command(mode, data_dir, output_dir, accelerator="gpu"):
    if mode == "audit":
        return [sys.executable, str(ROOT / "scripts/audit_completed_fivefold.py"),
                "--run-dir", str(output_dir), "--data-dir", str(data_dir)]
    shared = ["--data-dir", str(data_dir), "--output-dir", str(output_dir),
              "--models", "TFT", "--initial-reference", "10",
              "--input-window", "50", "--horizon", "10", "--folds", "5",
              "--seed", "42", "--torch-threads", "2", "--accelerator", accelerator,
              "--bootstrap-repetitions", "1000"]
    if mode == "check":
        return [sys.executable, str(ROOT / "scripts/run_paper_darts.py"),
                *shared, "--dry-run"]
    if mode == "train":
        return [sys.executable, str(ROOT / "scripts/tune_subjectwise_fivefold.py"),
                "--search-space", str(ROOT / "config/tft_candidates.json"),
                *shared]
    raise ValueError(f"Unknown mode: {mode}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "train", "audit"], default="check")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for check/train; completed run directory for audit.")
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="gpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"TFT mode={args.mode}; L0=10, W=50, H=10; five participant-level folds.", flush=True)
    command = build_command(args.mode, args.data_dir, args.output_dir, args.accelerator)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
