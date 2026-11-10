"""Choosing which N molecules of the train pool U get labels.

random: N molecules uniformly at random.
maxmin: farthest-point sampling in Tanimoto distance on ECFP4 (RDKit MaxMinPicker),
        started from one random molecule.
"""

from __future__ import annotations

import numpy as np
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

STRATEGIES = ("random", "maxmin")


def _rng(sample_seed: int, split_seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([sample_seed, split_seed, stream]))


def random_selection(pool: np.ndarray, n: int, sample_seed: int, split_seed: int = 0) -> np.ndarray:
    """Sorted indices of n random members of the (sorted) pool."""
    return np.sort(_rng(sample_seed, split_seed, 0).permutation(pool)[:n])


def _bit_vectors(fingerprints: np.ndarray) -> list:
    vectors = []
    for row in fingerprints:
        vector = DataStructs.ExplicitBitVect(fingerprints.shape[1])
        vector.SetBitsFromList(np.flatnonzero(row).tolist())
        vectors.append(vector)
    return vectors


def maxmin_selection(
    pool: np.ndarray, ecfp: np.ndarray, n: int, sample_seed: int, split_seed: int = 0
) -> np.ndarray:
    """Indices of n pool members in the order MaxMinPicker picks them."""
    first = int(_rng(sample_seed, split_seed, 2).integers(len(pool)))
    picks = MaxMinPicker().LazyBitVectorPick(
        _bit_vectors(ecfp[pool]), len(pool), n, firstPicks=(first,), seed=-1
    )
    return pool[np.asarray(list(picks), dtype=np.int64)]


def select(strategy, pool, ecfp, n, sample_seed, split_seed=0) -> np.ndarray:
    if strategy == "random":
        return random_selection(pool, n, sample_seed, split_seed)
    if strategy == "maxmin":
        return maxmin_selection(pool, ecfp, n, sample_seed, split_seed)
    raise ValueError(f"unknown strategy {strategy!r}")
