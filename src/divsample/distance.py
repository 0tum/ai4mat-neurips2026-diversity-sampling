"""Distance of a test molecule to the train pool U: the mean Tanimoto distance
(on ECFP4) to its k = 100 nearest molecules in U."""

from __future__ import annotations

import numpy as np

K_NEIGHBORS = 100


def knn_distance(
    ecfp: np.ndarray,
    pool: np.ndarray,
    queries: np.ndarray,
    k: int = K_NEIGHBORS,
    pool_chunk: int = 65_536,
    query_chunk: int = 2_048,
) -> np.ndarray:
    """Mean of (1 - Tanimoto similarity) over the k most similar pool molecules.

    Bit counts and intersections are integers below 2**24, so float32 arithmetic
    is exact and the result does not depend on the chunk sizes or the BLAS library.
    """
    best = np.full((len(queries), k), -np.inf, dtype=np.float32)
    query_bits = [np.asarray(ecfp[queries[s : s + query_chunk]], dtype=np.float32)
                  for s in range(0, len(queries), query_chunk)]
    for start in range(0, len(pool), pool_chunk):
        block = np.asarray(ecfp[pool[start : start + pool_chunk]], dtype=np.float32)
        block_count = block.sum(axis=1, dtype=np.float32)
        for q, query in enumerate(query_bits):
            rows = slice(q * query_chunk, q * query_chunk + len(query))
            similarity = query @ block.T
            union = query.sum(axis=1, dtype=np.float32)[:, None] + block_count[None, :] - similarity
            np.divide(similarity, union, out=similarity, where=union > 0)
            merged = np.concatenate([best[rows], similarity], axis=1)
            best[rows] = -np.partition(-merged, k - 1, axis=1)[:, :k]
    top = -np.sort(-best, axis=1)  # most similar first
    return (1.0 - top).mean(axis=1)
