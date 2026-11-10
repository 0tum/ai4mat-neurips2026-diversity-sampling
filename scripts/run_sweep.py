"""Select training sets, fit the models and save test-set predictions.

    uv run python scripts/run_sweep.py configs/qm9_gap.yaml
    uv run python scripts/run_sweep.py configs/qm9_gap.yaml --shard 0/4   # 1 of 4 parallel jobs

Finished runs are skipped. See src/divsample/sweep.py for the output layout.
"""

import argparse
from pathlib import Path

from divsample.sweep import load_config, run_sweep

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th run starting at i")
    args = parser.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))
    run_sweep(load_config(args.config), args.data_dir, args.results, (index, count))


if __name__ == "__main__":
    main()
