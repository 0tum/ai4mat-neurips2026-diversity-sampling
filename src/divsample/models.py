"""The four regressors. Each is fitted on the N selected molecules only.

krr       RBF kernel ridge, targets centered on the selected labels
rf        random forest
mlp       PyTorch MLP (64-32, ReLU), targets standardized on the selected labels
chemprop  Chemprop 2 D-MPNN, through its command line

rdkit2d inputs are standardized on the selected molecules (constant columns dropped).
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

# Required by torch.use_deterministic_algorithms on CUDA; must be set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODELS = ("krr", "rf", "mlp", "chemprop")
INPUTS = ("ecfp", "rdkit2d", "smiles")

DEFAULT_PARAMS = {
    "seed": 123,
    "krr": {"alpha": 1.0, "gamma": None},
    "rf": {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
    "mlp": {"hidden_layers": [64, 32], "weight_decay": 1e-4, "learning_rate": 1e-3,
            "epochs": 1000, "batch_size": 200},
    "chemprop": {"epochs": 50, "warmup_epochs": 2, "batch_size": 64},
}


def standardize_rdkit2d(train: np.ndarray, evaluations: dict[str, np.ndarray]):
    from sklearn.feature_selection import VarianceThreshold  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    selector, scaler = VarianceThreshold(0), StandardScaler()
    x = scaler.fit_transform(selector.fit_transform(np.asarray(train, dtype=np.float32)))
    transform = lambda v: scaler.transform(selector.transform(np.asarray(v, dtype=np.float32)))
    return (
        np.asarray(x, dtype=np.float32),
        {k: np.asarray(transform(v), dtype=np.float32) for k, v in evaluations.items()},
    )


# ---------------------------------------------------------------------- sklearn
def fit_predict_krr(x, y, evaluations, params, seed):
    from scipy.linalg import LinAlgWarning  # noqa: PLC0415
    from sklearn.kernel_ridge import KernelRidge  # noqa: PLC0415

    mean = y.mean()
    model = KernelRidge(kernel="rbf", alpha=params["alpha"], gamma=params["gamma"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        model.fit(x, y - mean)
    return {k: np.asarray(model.predict(v) + mean, dtype=np.float64) for k, v in evaluations.items()}


def fit_predict_rf(x, y, evaluations, params, seed):
    from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

    model = RandomForestRegressor(**params, n_jobs=-1, random_state=seed).fit(x, y)
    return {k: np.asarray(model.predict(v), dtype=np.float64) for k, v in evaluations.items()}


# -------------------------------------------------------------------------- MLP
def fit_predict_mlp(x, y, evaluations, params, seed):
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_mean = float(y.mean())
    y_scale = float(y.std()) or 1.0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    layers, width = [], x.shape[1]
    for hidden in params["hidden_layers"]:
        layers += [nn.Linear(width, hidden), nn.ReLU()]
        width = hidden
    model = nn.Sequential(*layers, nn.Linear(width, 1)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    all_x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    all_y = torch.from_numpy(np.asarray((y - y_mean) / y_scale, dtype=np.float32)).to(device)
    n = len(all_x)
    batch = min(params["batch_size"], n)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(params["epochs"]):
        model.train()
        # The same random draws as DataLoader(shuffle=True, generator=generator).
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(n, generator=generator)
        torch.randperm(n, generator=generator)
        order = order.to(device)
        for start in range(0, n, batch):
            rows = order[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(all_x[rows]).squeeze(-1), all_y[rows])
            loss.backward()
            optimizer.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for k, v in evaluations.items():
            pred = model(torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device))
            out[k] = np.asarray(pred.squeeze(-1).cpu().numpy() * y_scale + y_mean, dtype=np.float64)
    return out


# --------------------------------------------------------------------- Chemprop
def _write_smiles(path: Path, smiles, target=None) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["smiles"] if target is None else ["smiles", "target"])
        for i, smi in enumerate(smiles):
            writer.writerow([smi] if target is None else [smi, float(target[i])])


def fit_predict_chemprop(smiles, y, evaluations, params, seed):
    import pandas as pd  # noqa: PLC0415

    chemprop = [str(Path(sys.executable).with_name("chemprop"))]
    common = ["--smiles-columns", "smiles", "--batch-size", str(params["batch_size"]),
              "--num-workers", "0", "--accelerator", "auto", "--devices", "auto"]
    with tempfile.TemporaryDirectory(prefix="chemprop_") as tmp:
        work = Path(tmp)
        _write_smiles(work / "train.csv", smiles, y)
        subprocess.run(
            [*chemprop, "train", "--data-path", str(work / "train.csv"),
             "--output-dir", str(work / "model"), "--target-columns", "target",
             "--task-type", "regression", "--epochs", str(params["epochs"]),
             "--warmup-epochs", str(params["warmup_epochs"]),
             "--pytorch-seed", str(seed), "--data-seed", str(seed),
             "--split-sizes", "1", "0", "0", *common],
            check=True,
        )
        out = {}
        for k, v in evaluations.items():
            _write_smiles(work / f"{k}.csv", v)
            subprocess.run(
                [*chemprop, "predict", "--test-path", str(work / f"{k}.csv"),
                 "--output", str(work / f"{k}_pred.csv"), "--model-paths", str(work / "model"),
                 *common],
                check=True,
            )
            out[k] = pd.read_csv(work / f"{k}_pred.csv")["target"].to_numpy(dtype=np.float64)
    return out


FIT_PREDICT = {
    "krr": fit_predict_krr,
    "rf": fit_predict_rf,
    "mlp": fit_predict_mlp,
    "chemprop": fit_predict_chemprop,
}


def fit_predict(model, model_input, features, y, train, evaluations, params):
    """Fit on rows ``train`` and predict each array of rows in ``evaluations``.

    ``features`` maps an input name to an array with one entry per molecule.
    """
    seed = params["seed"]
    values = features[model_input]
    x = values[train]
    x_eval = {k: values[rows] for k, rows in evaluations.items()}
    if model_input == "ecfp":
        x = x.astype(np.float32)
        x_eval = {k: v.astype(np.float32) for k, v in x_eval.items()}
    elif model_input == "rdkit2d":
        x, x_eval = standardize_rdkit2d(x, x_eval)
    return FIT_PREDICT[model](x, y[train], x_eval, params[model], seed)
