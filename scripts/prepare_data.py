"""Build everything the experiments read, from the public raw files.

    uv run python scripts/prepare_data.py qm9
    uv run python scripts/prepare_data.py pcqm --workers 32

Writes, under <data-dir>/<dataset>/:
    molecules.csv                      cleaned molecules; the row order all indices refer to
    features/ecfp.npy, rdkit2d.npy     ECFP4 (bool) and RDKit 2D descriptors (float32)
    splits/<split>/                    train_pool / iid_test / ood_test row indices, and
                                       {iid,ood}_test_distance.npy (k=100 Tanimoto distance to U)
    checksums.json                     hashes of all arrays, compared with checksums/<dataset>.json

Each step is skipped when its output already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from divsample import data, distance, features, splits

REPO = Path(__file__).resolve().parents[1]


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}{array.shape}".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def build_features(dataset_dir: Path, smiles: np.ndarray, workers: int) -> None:
    out = dataset_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "ecfp.npy").exists():
        print("computing ECFP4")
        np.save(out / "ecfp.npy", features.ecfp(smiles))
    if not (out / "rdkit2d.npy").exists():
        print("computing RDKit 2D descriptors")
        values, names = features.rdkit2d(smiles, workers)
        np.save(out / "rdkit2d.npy", values)
        np.save(out / "rdkit2d_names.npy", names)


def split_definitions(dataset: str, molecules, ecfp: np.ndarray) -> dict:
    """Name -> function returning the split, evaluated only when the split is missing."""
    if dataset == "qm9":
        smiles = molecules["smiles"].to_numpy()
        definitions = {
            f"property_{p}": (lambda p=p: splits.property_split_qm9(molecules[p].to_numpy()))
            for p in data.QM9_PROPERTIES
        }
        definitions["scaffold"] = lambda: splits.scaffold_split(smiles, seed=0)
        definitions["cluster"] = lambda: splits.cluster_split(
            splits.cluster_labels(ecfp, seed=0), leave_out=0, seed=0
        )
        return definitions
    full = {}

    def pool_1p5m():
        if "split" not in full:
            full["split"] = splits.property_split_pcqm(molecules["gap"].to_numpy())
        return full["split"]

    return {
        "property_gap_1.5M": pool_1p5m,
        "property_gap_150k": lambda: splits.downsample_pool(pool_1p5m(), 150_000, seed=42),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["qm9", "pcqm"])
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    dataset_dir = args.data_dir / args.dataset

    if not (dataset_dir / "molecules.csv").exists():
        if args.dataset == "qm9":
            data.build_qm9(dataset_dir)
        else:
            data.build_pcqm(dataset_dir, args.workers)
    molecules = data.load_molecules(dataset_dir)
    build_features(dataset_dir, molecules["smiles"].to_numpy(), args.workers)
    ecfp = np.load(dataset_dir / "features" / "ecfp.npy")

    sums = {
        "molecules.csv": data.sha256_file(dataset_dir / "molecules.csv"),
        "features/ecfp.npy": array_sha256(ecfp),
        "features/rdkit2d.npy": array_sha256(np.load(dataset_dir / "features" / "rdkit2d.npy")),
    }
    for name, make in split_definitions(args.dataset, molecules, ecfp).items():
        split_dir = dataset_dir / "splits" / name
        if not (split_dir / "ood_test_distance.npy").exists():
            print(f"split {name}")
            split = make()
            distances = {
                test: distance.knn_distance(ecfp, split["train_pool"], split[test])
                for test in splits.TESTS
            }
            splits.save_split(split_dir, split, distances)
        split, distances = splits.load_split(split_dir)
        sizes = ", ".join(f"{k}={len(v)}" for k, v in split.items())
        print(f"{name}: {sizes}")
        for key, values in {**split, **{f"{k}_distance": v for k, v in distances.items()}}.items():
            sums[f"splits/{name}/{key}.npy"] = array_sha256(values)

    (dataset_dir / "checksums.json").write_text(json.dumps(sums, indent=1) + "\n")
    expected_path = REPO / "checksums" / f"{args.dataset}.json"
    if not expected_path.exists():
        print(f"no reference checksums at {expected_path}")
        return
    expected = json.loads(expected_path.read_text())
    differ = [key for key in expected if sums.get(key) != expected[key]]
    if differ:
        print("differs from the paper's data (see README, 'Checking the data'):")
        for key in differ:
            print(f"  {key}")
    else:
        print(f"all {len(expected)} checksums match the paper's data")


if __name__ == "__main__":
    main()
