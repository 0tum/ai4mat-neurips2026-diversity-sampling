"""Error ratio r and label ratio M/N as a function of distance to the train pool.

Test molecules of each split are binned into deciles of their k-NN Tanimoto
distance to U (decile 0 = closest). For each condition (model + input), seed
and decile:

    r   = MAE_MaxMin(N) / MAE_Random(N)
    M/N where M is the number of random labels that reaches MAE_MaxMin(N), read off
        the seed's random learning curve over the training sizes of the sweep by
        linear interpolation in log(N)-log(MAE). The curve is first replaced by its
        running minimum. M outside the grid is censored at the grid end.

Summaries average log r and log2 M/N over conditions within each seed, then report
the mean over seeds and the standard deviation across seeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import splits
from .sweep import Run, run_dir

N_BINS = 10


def distance_deciles(distance: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Decile of each test molecule; equal-count bins in order of distance."""
    decile = np.empty(len(distance), dtype=int)
    for i, rows in enumerate(np.array_split(np.argsort(distance), n_bins)):
        decile[rows] = i
    return decile


def decile_mae(path: Path, decile: np.ndarray, molecule_ids: np.ndarray) -> pd.Series:
    frame = pd.read_csv(path, dtype={"molecule_id": str}, float_precision="round_trip")
    if not np.array_equal(frame["molecule_id"].to_numpy(), molecule_ids):
        raise ValueError(f"{path}: test molecules differ from the split")
    error = np.abs(frame["prediction"].to_numpy() - frame["target"].to_numpy())
    return pd.Series(error).groupby(decile).mean()


def log2_m_over_n(random_curve: np.ndarray, mae_maxmin: float, n: int, grid) -> tuple[float, str]:
    """log2(M/N) and whether M lies within the grid ('within') or is censored ('low'/'high')."""
    log_grid = np.log10(np.asarray(grid, dtype=float))
    curve = np.minimum.accumulate(np.log10(random_curve))
    target = np.log10(mae_maxmin)
    if target > curve[0]:
        log_m, side = log_grid[0], "low"
    elif target < curve[-1]:
        log_m, side = log_grid[-1], "high"
    else:
        log_m, side = np.interp(target, curve[::-1], log_grid[::-1]), "within"
    return (log_m - np.log10(n)) / np.log10(2.0), side


def condition_cells(results, config, split, condition, test, decile, ids, n, seeds) -> pd.DataFrame:
    model, model_input = condition.split("_", 1)
    grid = sorted(config["train_sizes"])

    def mae(strategy, size, seed):
        path = run_dir(results, config, Run(split, model, model_input, size, seed, strategy))
        return decile_mae(path / f"{test}.csv.gz", decile, ids)

    rows = []
    for seed in seeds:
        maxmin = mae("maxmin", n, seed)
        curve = pd.DataFrame({size: mae("random", size, seed) for size in grid})
        for d in maxmin.index:
            log2_mn, side = log2_m_over_n(curve.loc[d].to_numpy(), maxmin[d], n, grid)
            rows.append({
                "decile": d, "seed": seed,
                "mae_random": curve.loc[d, n], "mae_maxmin": maxmin[d],
                "log_r": np.log(maxmin[d] / curve.loc[d, n]),
                "log2_mn": log2_mn, "mn_censored": side,
            })
    return pd.DataFrame(rows)


def missing_runs(results, config, split, condition, n, seeds) -> list[Path]:
    model, model_input = condition.split("_", 1)
    wanted = [(s, n, "maxmin") for s in seeds]
    wanted += [(s, size, "random") for s in seeds for size in config["train_sizes"]]
    paths = [run_dir(results, config, Run(split, model, model_input, size, s, st))
             for s, size, st in wanted]
    return [p for p in paths if not (p / "ood_test.csv.gz").exists()]


def compute(config, data_dir: Path, results: Path, n: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell table (condition x seed x decile) and the summary per split, test set and decile."""
    dataset_dir = Path(data_dir) / config["dataset"]
    ids_all = pd.read_csv(dataset_dir / "molecules.csv", usecols=["mol_id"], dtype=str)["mol_id"].to_numpy()
    seeds = list(config["seeds"])
    cells = []
    for split in config["splits"]:
        indices, distances = splits.load_split(dataset_dir / "splits" / split)
        complete = []
        for condition in config["conditions"]:
            missing = missing_runs(results, config, split, condition, n, seeds)
            if missing:
                print(f"{split} {condition}: left out, {len(missing)} runs missing (e.g. {missing[0]})")
            else:
                complete.append(condition)
        for test in splits.TESTS:
            decile = distance_deciles(distances[test])
            median = pd.Series(distances[test]).groupby(decile).median()
            size = pd.Series(decile).value_counts().sort_index()
            for condition in complete:
                frame = condition_cells(results, config, split, condition, test, decile,
                                        ids_all[indices[test]], n, seeds)
                cells.append(frame.assign(
                    split=split, test=test, condition=condition,
                    distance=frame["decile"].map(median), n_molecules=frame["decile"].map(size),
                ))
    cells = pd.concat(cells, ignore_index=True)
    cells = cells[["split", "test", "condition", "decile", "distance", "n_molecules", "seed",
                   "mae_random", "mae_maxmin", "log_r", "log2_mn", "mn_censored"]]
    return cells, summarize(cells)


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["split", "test", "decile"]
    by_seed = cells.assign(
        low=cells["mn_censored"].eq("low"), high=cells["mn_censored"].eq("high")
    ).groupby([*keys, "seed"]).agg(
        log_r=("log_r", "mean"), log2_mn=("log2_mn", "mean"),
        distance=("distance", "first"), n_molecules=("n_molecules", "first"),
        n_conditions=("condition", "nunique"), low=("low", "sum"), high=("high", "sum"),
    )
    grouped = by_seed.groupby(keys)
    summary = grouped.agg(
        distance=("distance", "first"), n_molecules=("n_molecules", "first"),
        n_conditions=("n_conditions", "max"), n_seeds=("log_r", "size"),
        log_r=("log_r", "mean"), log_r_seed_sd=("log_r", "std"),
        log2_mn=("log2_mn", "mean"), log2_mn_seed_sd=("log2_mn", "std"),
        log2_mn_seed_min=("log2_mn", "min"), log2_mn_seed_max=("log2_mn", "max"),
    )
    summary["n_seeds_censored_low"] = grouped["low"].apply(lambda s: int((s > 0).sum()))
    summary["n_seeds_censored_high"] = grouped["high"].apply(lambda s: int((s > 0).sum()))
    summary["r"] = np.exp(summary["log_r"])
    summary["m_over_n"] = np.exp2(summary["log2_mn"])
    return summary.reset_index()
