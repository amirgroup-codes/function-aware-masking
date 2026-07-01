"""CLI entry point: load config(s), build components, train.

Two usage modes:

  1. Multi-model shorthand (preferred for the 9-model comparison run).
     Pass one or more model flags; they train sequentially in command-line
     order, with each model's stdout/stderr/logging redirected to
     ``logs/<name>.log`` automatically. The terminal stays clean and shows
     only short status banners between models.

         python scripts/train.py --uniform --cdr
         python scripts/train.py --multispecific --hybrid_paired

  2. Legacy single-config path (no log redirection):

         python scripts/train.py --config configs/medium.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import load_config
from training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Maps each --<flag> to (config_path, log/experiment name). The log name
# matches the experiment key in configs/experiments.yaml so the file
# layout under logs/ mirrors the eval pipeline's outputs.
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "uniform":           ("configs/medium.yaml",                   "uniform_medium"),
    "cdr":               ("configs/cdr_medium.yaml",               "cdr_medium"),
    "span":              ("configs/span_medium.yaml",              "span_medium"),
    "structure":         ("configs/structure_medium.yaml",         "structure_medium"),
    "interface":         ("configs/interface_medium.yaml",         "interface_medium"),
    "germline":          ("configs/germline_medium.yaml",          "germline_medium"),
    "multispecific":     ("configs/multispecific_medium.yaml",     "multispecific_medium"),
    "hybrid_curriculum": ("configs/hybrid_curriculum_medium.yaml", "hybrid_curriculum_medium"),
    "hybrid_paired":     ("configs/hybrid_paired_medium.yaml",     "hybrid_paired_medium"),
    # Post-paper optimizations introduced after the structure-prior
    # upgrade (IgFold real coordinates). Each variant tests a separate
    # design axis; see the yaml header for the specific motivation.
    "structure_longrange":  ("configs/structure_longrange_medium.yaml",  "structure_longrange_medium"),
    "intersection":         ("configs/intersection_medium.yaml",         "intersection_medium"),
    "hybrid_stretched":     ("configs/hybrid_stretched_medium.yaml",     "hybrid_stretched_medium"),
    "hybrid_reverse":       ("configs/hybrid_reverse_medium.yaml",       "hybrid_reverse_medium"),
    "hybrid_warmstart":     ("configs/hybrid_warmstart_medium.yaml",     "hybrid_warmstart_medium"),
    "hybrid_intersection":  ("configs/hybrid_intersection_medium.yaml",  "hybrid_intersection_medium"),
    "hybrid_perbatch":      ("configs/hybrid_perbatch_medium.yaml",      "hybrid_perbatch_medium"),
    "hybrid_weighted":      ("configs/hybrid_weighted_medium.yaml",      "hybrid_weighted_medium"),
    "hybrid_adaptive":      ("configs/hybrid_adaptive_medium.yaml",      "hybrid_adaptive_medium"),
}


@contextmanager
def _redirect_fds_to_log(log_path: Path):
    """Redirect file descriptors 1 and 2 to ``log_path`` for the block.

    Operating at the FD level (rather than reassigning ``sys.stdout``)
    captures EVERYTHING: print(), logging, tqdm progress bars, transformers
    warnings, torch C-extension prints — anything that ends up writing to
    stdout/stderr through any path lands in the file. tqdm auto-detects
    that stderr is no longer a tty and switches to ``\\n``-terminated
    output, which is exactly what we want for ``tail -f``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()

    saved_out = os.dup(1)
    saved_err = os.dup(2)
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
    finally:
        os.close(log_fd)

    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)


def _train_one(flag: str, log_dir: Path) -> None:
    """Train one model (by flag), redirecting its output to a log file."""
    config_path, name = MODEL_REGISTRY[flag]
    log_path = log_dir / f"{name}.log"

    # Banner goes to the original terminal so the user (or tmux pane) can
    # see which model is currently active without tailing the log file.
    print(f"=== [{flag}] training {name}", flush=True)
    print(f"    config: {config_path}", flush=True)
    print(f"    log:    {log_path}", flush=True)

    with _redirect_fds_to_log(log_path):
        config = load_config(config_path)
        ckpt = train(config)
        print(f"\nCheckpoint saved to: {ckpt}", flush=True)

    print(f"=== [{flag}] done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one or more antibody MLM models sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train.py --uniform\n"
            "  python scripts/train.py --uniform --cdr        "
            "# uniform first, then cdr\n"
            "  python scripts/train.py --config configs/medium.yaml  "
            "# legacy single-config mode\n\n"
            "Each --<flag> writes to logs/<name>_medium.log automatically.\n"
            "Order on the command line is preserved.\n"
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a single experiment config YAML (legacy single-model mode).",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for per-model log files (default: logs).",
    )
    for flag in MODEL_REGISTRY:
        parser.add_argument(
            f"--{flag}", action="store_true",
            help=f"Train {MODEL_REGISTRY[flag][1]}",
        )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Preserve command-line order so `--cdr --uniform` runs cdr first.
    # argparse stores booleans in args but loses ordering, so re-scan
    # sys.argv directly.
    selected: list[str] = []
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            key = tok[2:]
            if key in MODEL_REGISTRY and key not in selected:
                selected.append(key)

    if args.config and selected:
        parser.error("Pass either --config or one or more model flags, not both.")

    if args.config:
        # Legacy path: don't redirect, let output stream to the terminal.
        config = load_config(args.config)
        ckpt = train(config)
        print(f"\nTraining complete. Checkpoint saved to: {ckpt}")
        return

    if not selected:
        parser.error(
            "No models selected. Pass --config <yaml> or one or more model "
            "flags (e.g. --uniform --cdr). See --help for the full list."
        )

    print(f"Sequential training queue: {' -> '.join(selected)}")
    failed: list[str] = []
    for flag in selected:
        try:
            _train_one(flag, log_dir)
        except Exception as e:
            # Don't let one model's failure cancel the rest of the queue.
            failed.append(flag)
            print(f"!!! [{flag}] FAILED: {e}", flush=True)
            print(
                f"    See {log_dir / (MODEL_REGISTRY[flag][1] + '.log')} for details",
                flush=True,
            )
            print("    Continuing to next model in queue", flush=True)

    if failed:
        print(f"\nDone, with {len(failed)} failure(s): {', '.join(failed)}")
        sys.exit(1)
    print("\nAll models complete.")


if __name__ == "__main__":
    main()
