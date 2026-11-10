"""Run every (split, model, N, seed, strategy) combination of a config.

Each run writes
    <results>/<dataset>_<target>/<split>/<model>_<input>/n<N>/seed<s>/<strategy>/
        train_indices.npy      selected molecules (rows of molecules.csv), in pick order
        iid_test.csv.gz        molecule_id, target, prediction
        ood_test.csv.gz
A run whose two prediction files exist is skipped, so an interrupted sweep can be restarted.
"""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import data, models, selection, splits


@dataclass(frozen=True)
class Run:
    split: str
    model: str
    model_input: str
    n: int
    seed: int
    strategy: str


def load_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    params = copy.deepcopy(models.DEFAULT_PARAMS)
    for key, value in (config.get("params") or {}).items():
        params[key] = {**params[key], **value} if isinstance(value, dict) else value
    config["params"] = params
    return config


def experiment_dir(results: Path, config: dict) -> Path:
    return Path(results) / f"{config['dataset']}_{config['target']}"


def run_dir(results: Path, config: dict, run: Run) -> Path:
    return (experiment_dir(results, config) / run.split / f"{run.model}_{run.model_input}"
            / f"n{run.n}" / f"seed{run.seed}" / run.strategy)


def all_runs(config: dict) -> list[Run]:
    conditions = [c.split("_", 1) for c in config["conditions"]]
    return [
        Run(split, model, model_input, n, seed, strategy)
        for split, (model, model_input), n, seed, strategy in itertools.product(
            config["splits"], conditions, config["train_sizes"], config["seeds"],
            selection.STRATEGIES,
        )
    ]


def is_done(path: Path) -> bool:
    return all((path / f"{test}.csv.gz").exists() for test in splits.TESTS)


def run_sweep(config: dict, data_dir: Path, results: Path, shard: tuple[int, int] = (0, 1)) -> None:
    dataset_dir = Path(data_dir) / config["dataset"]
    molecules = data.load_molecules(dataset_dir)
    y = molecules[config["target"]].to_numpy(dtype=np.float64)
    ecfp = np.load(dataset_dir / "features" / "ecfp.npy")
    features = {"ecfp": ecfp, "smiles": molecules["smiles"].to_numpy(dtype=str)}
    if any(c.endswith("_rdkit2d") for c in config["conditions"]):
        features["rdkit2d"] = np.load(dataset_dir / "features" / "rdkit2d.npy")

    index, count = shard
    todo = [r for i, r in enumerate(all_runs(config)) if i % count == index]
    loaded = {}
    for i, run in enumerate(todo, start=1):
        out = run_dir(results, config, run)
        if is_done(out):
            continue
        if run.split not in loaded:
            loaded[run.split] = splits.load_split(dataset_dir / "splits" / run.split)[0]
        split = loaded[run.split]
        started = time.perf_counter()
        train = selection.select(run.strategy, split["train_pool"], ecfp, run.n, run.seed)
        evaluations = {test: split[test] for test in splits.TESTS}
        predictions = models.fit_predict(
            run.model, run.model_input, features, y, train, evaluations, config["params"]
        )
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "train_indices.npy", train)
        for test, rows in evaluations.items():
            pd.DataFrame({
                "molecule_id": molecules["mol_id"].to_numpy()[rows],
                "target": y[rows],
                "prediction": predictions[test],
            }).to_csv(out / f"{test}.csv.gz", index=False)
        print(f"[{i}/{len(todo)}] {out} ({time.perf_counter() - started:.0f} s)", flush=True)
