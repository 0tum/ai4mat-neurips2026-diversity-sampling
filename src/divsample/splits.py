"""Train pool (U), IID test and OOD test for the three definitions of extrapolation.

All indices are row positions in ``molecules.csv`` and are returned sorted.
The paper uses split seed 0 for scaffold and cluster, and cluster 0 as the
left-out cluster.
"""

from __future__ import annotations

import math
import random

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

RDLogger.DisableLog("rdApp.*")

Split = dict[str, np.ndarray]  # keys: train_pool, iid_test, ood_test


def _as_split(train_pool, iid_test, ood_test) -> Split:
    split = {
        "train_pool": np.sort(np.asarray(train_pool, dtype=np.int64)),
        "iid_test": np.sort(np.asarray(iid_test, dtype=np.int64)),
        "ood_test": np.sort(np.asarray(ood_test, dtype=np.int64)),
    }
    joined = np.concatenate(list(split.values()))
    if len(np.unique(joined)) != len(joined):
        raise ValueError("split partitions overlap")
    return split


# ---------------------------------------------------------------------- property
def property_split_qm9(
    y: np.ndarray, n_ood: int = 10_000, iid_fraction: float = 0.1, seed: int = 42
) -> Split:
    """OOD = the n_ood molecules with the lowest KDE density of the property (BOOM).
    The rest goes to IID test with probability iid_fraction, otherwise to U."""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    density = np.exp(KernelDensity(kernel="gaussian", bandwidth="scott").fit(y).score_samples(y))
    ood = set(np.argpartition(density, n_ood)[:n_ood].tolist())
    rng = random.Random(seed)
    train, iid = [], []
    for i in range(len(y)):
        if i in ood:
            continue
        (iid if rng.random() < iid_fraction else train).append(i)
    return _as_split(train, iid, sorted(ood))


def fft_kde_density(values: np.ndarray, gridsize: int = 2**18) -> np.ndarray:
    """Gaussian KDE (Scott's rule) on a grid via FFT, read back at each value."""
    bandwidth = float(np.std(values, ddof=1)) * len(values) ** (-1.0 / 5.0)
    lower = float(np.min(values) - 4.0 * bandwidth)
    upper = float(np.max(values) + 4.0 * bandwidth)
    grid = np.linspace(lower, upper, gridsize, dtype=np.float64)
    dx = float(grid[1] - grid[0])
    counts, _ = np.histogram(values, bins=gridsize, range=(lower, upper))
    offsets = np.arange(-gridsize + 1, gridsize, dtype=np.float64) * dx
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    kernel /= kernel.sum()
    size = 1 << ((len(counts) + len(kernel) - 2).bit_length())
    density = np.fft.irfft(np.fft.rfft(counts, size) * np.fft.rfft(kernel, size), size)
    start = len(kernel) // 2
    density = density[start : start + len(counts)] / (len(values) * dx)
    return np.interp(values, grid, density)


def property_split_pcqm(
    y: np.ndarray, n_iid: int = 10_000, n_ood: int = 10_000, seed: int = 42
) -> Split:
    """OOD = the n_ood lowest-density molecules; IID test = n_iid drawn from the rest."""
    y = np.asarray(y, dtype=np.float64)
    ood = np.argpartition(fft_kde_density(y), n_ood - 1)[:n_ood]
    is_ood = np.zeros(len(y), dtype=bool)
    is_ood[ood] = True
    iid = sorted(random.Random(seed).sample(np.flatnonzero(~is_ood).tolist(), n_iid))
    is_test = is_ood.copy()
    is_test[iid] = True
    return _as_split(np.flatnonzero(~is_test), iid, ood)


def downsample_pool(split: Split, size: int = 150_000, seed: int = 42) -> Split:
    """Smaller U with the same IID and OOD test sets."""
    pool = split["train_pool"]
    keep = sorted(random.Random(seed).sample(range(len(pool)), size))
    return _as_split(pool[keep], split["iid_test"], split["ood_test"])


# ---------------------------------------------------------------------- scaffold
def _murcko(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES, or None if the scaffold has no ring."""
    mol = Chem.MolFromSmiles(smiles)
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core.GetNumAtoms() == 0 or core.GetRingInfo().NumRings() == 0:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles: np.ndarray, seed: int = 0, test_fraction: float = 0.2) -> Split:
    """Among molecules with a cyclic scaffold, draw test_fraction at random as test.
    Test molecules whose scaffold also occurs in U are IID, the others OOD.
    Molecules without a ring scaffold are left out entirely."""
    scaffolds = [_murcko(str(s)) for s in smiles]
    cyclic = np.asarray([i for i, s in enumerate(scaffolds) if s is not None], dtype=np.int64)
    order = list(range(len(cyclic)))
    random.Random(seed).shuffle(order)
    n_test = math.ceil(len(cyclic) * test_fraction)
    test, train = cyclic[order[:n_test]], cyclic[order[n_test:]]
    train_scaffolds = {scaffolds[i] for i in train}
    iid = [i for i in test if scaffolds[i] in train_scaffolds]
    ood = [i for i in test if scaffolds[i] not in train_scaffolds]
    return _as_split(train, iid, ood)


# ----------------------------------------------------------------------- cluster
def cluster_labels(ecfp: np.ndarray, seed: int = 0, n_clusters: int = 10) -> np.ndarray:
    """ECFP -> PCA(64) -> k-means."""
    reduced = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit_transform(
        np.asarray(ecfp, dtype=np.float32)
    ).astype(np.float32)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300).fit_predict(
        reduced
    )


def cluster_split(labels: np.ndarray, leave_out: int = 0, seed: int = 0) -> Split:
    """OOD = one left-out cluster. IID test = a random draw from the other clusters,
    as large as the mean cluster."""
    rows = np.arange(len(labels), dtype=np.int64)
    ood, rest = rows[labels == leave_out], rows[labels != leave_out]
    n_iid = int(round(float(np.mean(np.bincount(labels)))))
    picked = np.random.default_rng(seed * 1000 + leave_out).choice(len(rest), n_iid, replace=False)
    is_iid = np.zeros(len(rest), dtype=bool)
    is_iid[picked] = True
    return _as_split(rest[~is_iid], rest[is_iid], ood)


PARTS = ("train_pool", "iid_test", "ood_test")
TESTS = ("iid_test", "ood_test")


def save_split(split_dir, split: Split, distances: dict[str, np.ndarray]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in PARTS:
        np.save(split_dir / f"{name}.npy", split[name])
    for name in TESTS:
        np.save(split_dir / f"{name}_distance.npy", distances[name])


def load_split(split_dir) -> tuple[Split, dict[str, np.ndarray]]:
    split = {name: np.load(split_dir / f"{name}.npy") for name in PARTS}
    distances = {name: np.load(split_dir / f"{name}_distance.npy") for name in TESTS}
    return split, distances
