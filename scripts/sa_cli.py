#!/usr/bin/env python3
"""
Simple CLI for running Figure 5 split-mode experiments.

Examples:
    python scripts/sa_cli.py fig5-split --trials 100 --epochs 500 --show
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root on sys.path to import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.figure5 import Figure5Config, run_figure5_experiments_split_modes  # type: ignore  # noqa: E402


def _parse_int_list(raw: str) -> tuple[int, ...]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return tuple(int(v) for v in vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seq-Attractor: experiments CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--trials", type=int, default=100, help="Number of trials per point")
    common.add_argument("--epochs", type=int, default=500, help="Training epochs")
    common.add_argument("--show", action="store_true", help="Show plots interactively")
    common.add_argument("--out", type=str, default="", help="Output directory (default: store/figure5)")
    common.add_argument("--no-timestamp", action="store_true", help="Do not create timestamped subdir")
    common.add_argument("--T-values", type=_parse_int_list, default=(10, 30, 50, 70, 90, 110, 140),
                        help="Comma-separated list, e.g. 10,30,50,70")
    common.add_argument("--Nh-values", type=_parse_int_list, default=(100, 325, 550, 775, 1000),
                        help="Comma-separated list, e.g. 100,325,550,775,1000")
    common.add_argument("--progress", action="store_true", help="Display tqdm progress bars")
    common.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes (-1=75%% cores, 0=all cores, default=1)",
    )

    p_split = sub.add_parser(
        "fig5-split",
        parents=[common],
        help="Run split-mode Figure 5 (compare V-only vs U+V for both scans)",
    )

    args = parser.parse_args()

    if args.command == "fig5-split":
        output_dir = Path(args.out) if args.out else None
        cfg = Figure5Config(
            num_trials=int(args.trials),
            noise_num=10,
            num_epochs=int(args.epochs),
            T_values=tuple(int(v) for v in args.T_values),
            N_h_values=tuple(int(v) for v in args.Nh_values),
        )
        run_figure5_experiments_split_modes(
            config=cfg,
            output_dir=output_dir,
            create_timestamp_dir=not bool(args.no_timestamp),
            show_images=bool(args.show),
            use_progress=bool(args.progress),
            workers=int(args.workers),
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()


