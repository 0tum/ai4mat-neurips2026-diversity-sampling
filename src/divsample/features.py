"""ECFP4 bit vectors (2048 bits) and RDKit 2D descriptors (217, float32)."""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ECFP_RADIUS = 2
ECFP_NBITS = 2048
DESCRIPTORS = tuple(Descriptors._descList)
N_DESCRIPTORS = 217  # RDKit 2026.03.3


def ecfp(smiles: np.ndarray, radius: int = ECFP_RADIUS, nbits: int = ECFP_NBITS) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    out = np.zeros((len(smiles), nbits), dtype=bool)
    for i, smi in enumerate(smiles):
        out[i] = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(str(smi))).astype(bool)
    return out


def rdkit2d_row(mol: Chem.Mol) -> np.ndarray:
    """All RDKit 2D descriptors; raises ValueError unless every value is finite in float32."""
    if len(DESCRIPTORS) != N_DESCRIPTORS:
        raise ValueError(f"expected {N_DESCRIPTORS} RDKit descriptors, got {len(DESCRIPTORS)}")
    row = np.empty(len(DESCRIPTORS), dtype=np.float64)
    for j, (name, function) in enumerate(DESCRIPTORS):
        try:
            row[j] = float(function(mol))
        except Exception as error:
            raise ValueError(f"descriptor {name} failed") from error
    row = row.astype(np.float32)
    if not np.isfinite(row).all():
        raise ValueError("non-finite descriptor")
    return row


def _rdkit2d_from_smiles(smiles: str) -> np.ndarray:
    return rdkit2d_row(Chem.MolFromSmiles(smiles))


def rdkit2d(smiles: np.ndarray, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    with Pool(workers) as pool:
        rows = pool.map(_rdkit2d_from_smiles, [str(s) for s in smiles], chunksize=1000)
    names = np.asarray([name for name, _ in DESCRIPTORS])
    return np.stack(rows).astype(np.float32), names
