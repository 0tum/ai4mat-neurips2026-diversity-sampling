"""Compute the error ratio r and the label ratio M/N per distance decile.

    uv run python scripts/compute_metrics.py configs/qm9_gap.yaml            # N = 1000
    uv run python scripts/compute_metrics.py configs/qm9_gap.yaml --n 3000

Writes results/<dataset>_<target>/metrics_n<N>_{cells,summary}.csv:
    cells    one row per split, test set, condition, seed and decile
    summary  one row per split, test set and decile: r and M/N averaged over
             conditions, mean and SD over seeds (log scale), and censoring counts
"""

import argparse
from pathlib import Path

from divsample.metrics import compute
from divsample.sweep import experiment_dir, load_config

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--n", type=int, default=1000, help="training size of MaxMin")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--results", type=Path, default=REPO / "results")
    args = parser.parse_args()
    config = load_config(args.config)
    cells, summary = compute(config, args.data_dir, args.results, args.n)
    out = experiment_dir(args.results, config)
    cells.to_csv(out / f"metrics_n{args.n}_cells.csv", index=False)
    summary.to_csv(out / f"metrics_n{args.n}_summary.csv", index=False)
    columns = ["split", "test", "decile", "distance", "r", "m_over_n"]
    print(summary[columns].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
