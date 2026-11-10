"""Download QM9 and PCQM4Mv2 and clean them into one molecule table per dataset.

The output, ``<data_dir>/<dataset>/molecules.csv``, fixes the row order that every
other array (features, split indices, distances) refers to.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import random
import re
import tarfile
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

from rdkit import Chem, RDLogger

from .features import rdkit2d_row

RDLogger.DisableLog("rdApp.*")

QM9_PROPERTIES = ("mu", "alpha", "gap")

# url, sha256 of the downloaded file
QM9_FILES = {
    "dsgdb9nsd.xyz.tar.bz2": (
        "https://figshare.com/ndownloader/files/3195389",
        "3a63848ac80691bdb8d41834b575afad345b9300d7a2db0c38adb7f6eaa8360c",
    ),
    "uncharacterized.txt": (
        "https://ndownloader.figshare.com/files/3195404",
        "3aa5115d540b356de94791d4a74c3bf1ed91c469ecf52a4f5d7cc0506fe02e24",
    ),
    "qm9.zip": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip",
        None,  # the archive is not versioned; its gdb9.sdf.csv is checked instead
    ),
}
QM9_PROPERTY_CSV_SHA256 = "73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb"

PCQM_FILES = {
    "pcqm4m-v2.zip": (
        "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip",
        "628ae612a3de752160929d93d1584a75257ae156e7b13d91b133d8db62e6d701",
    ),
}
PCQM_SAMPLE_SIZE = 1_500_000
PCQM_SAMPLE_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(raw_dir: Path, files: dict[str, tuple[str, str | None]]) -> None:
    """Download missing files and check the hash of every file.

    A file already placed in ``raw_dir`` (for example, downloaded by hand when a
    host refuses scripted downloads) is used as is.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in files.items():
        path = raw_dir / name
        if not path.exists():
            print(f"downloading {url} -> {path}")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            partial = path.with_name(path.name + ".part")
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                while block := response.read(1 << 20):
                    out.write(block)
            partial.rename(path)
        if expected is not None and sha256_file(path) != expected:
            raise ValueError(f"{path}: sha256 mismatch; delete it and download again")


def write_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------- QM9
XYZ_NAME = re.compile(r"dsgdb9nsd_(\d{6})\.xyz$")


def qm9_smiles_from_xyz(xyz_tar: Path) -> dict[str, str]:
    """SMILES stored in the official XYZ files (not inferred from coordinates)."""
    smiles = {}
    with tarfile.open(xyz_tar, mode="r:bz2") as tar:
        for member in tar:
            match = XYZ_NAME.search(member.name)
            if not member.isfile() or match is None:
                continue
            lines = tar.extractfile(member).read().decode().splitlines()
            n_atoms = int(lines[0])
            smiles[f"gdb_{int(match.group(1))}"] = lines[2 + n_atoms + 1].split()[0]
    return smiles


def qm9_uncharacterized(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0][0].isdigit():
            ids.add(f"gdb_{int(fields[0])}")
    return ids


def parses(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False


def build_qm9(dataset_dir: Path) -> Path:
    """Join SMILES with properties, drop uncharacterized molecules and RDKit failures."""
    raw = dataset_dir / "raw"
    fetch(raw, QM9_FILES)
    with zipfile.ZipFile(raw / "qm9.zip") as archive:
        property_csv = archive.read("gdb9.sdf.csv")
    if hashlib.sha256(property_csv).hexdigest() != QM9_PROPERTY_CSV_SHA256:
        raise ValueError("qm9.zip: gdb9.sdf.csv differs from the version used in the paper")

    smiles = qm9_smiles_from_xyz(raw / "dsgdb9nsd.xyz.tar.bz2")
    excluded = qm9_uncharacterized(raw / "uncharacterized.txt")
    rows = []
    for row in csv.DictReader(io.StringIO(property_csv.decode())):
        mol_id = row["mol_id"]
        if mol_id in excluded or not parses(smiles[mol_id]):
            continue
        rows.append([mol_id, smiles[mol_id], *(row[p] for p in QM9_PROPERTIES)])
    out = dataset_dir / "molecules.csv"
    write_table(out, ["mol_id", "smiles", *QM9_PROPERTIES], rows)
    print(f"QM9: {len(rows)} molecules -> {out}")
    return out


# --------------------------------------------------------------------------- PCQM
def pcqm_rows(zip_path: Path) -> list[tuple[str, str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        data = gzip.decompress(archive.read("pcqm4m-v2/raw/data.csv.gz")).decode()
    return [
        (f"pcqm_{int(row['idx'])}", row["smiles"], row["homolumogap"])
        for row in csv.DictReader(io.StringIO(data))
    ]


def _pcqm_check(item: tuple[str, str, str]) -> tuple[str | None, bool]:
    """Canonical SMILES (None if RDKit fails) and whether all RDKit 2D descriptors are finite."""
    _, smiles, gap = item
    if not gap:
        return None, False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        mol = None
    if mol is None:
        return None, False
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        rdkit2d_row(mol)
    except ValueError:
        return canonical, False
    return canonical, True


def build_pcqm(dataset_dir: Path, workers: int = 8) -> Path:
    """Keep molecules with a gap value that RDKit parses and describes, drop canonical
    SMILES that carry conflicting gap values, then sample 1.5M molecules in source order."""
    raw = dataset_dir / "raw"
    fetch(raw, PCQM_FILES)
    rows = pcqm_rows(raw / "pcqm4m-v2.zip")
    with Pool(workers) as pool:
        checks = pool.map(_pcqm_check, rows, chunksize=2000)

    first_gap: dict[str, str] = {}
    conflicting = set()
    for (_, _, gap), (canonical, _) in zip(rows, checks):
        if canonical is None:
            continue
        if first_gap.setdefault(canonical, gap) != gap:
            conflicting.add(canonical)
    kept = [
        row
        for row, (canonical, described) in zip(rows, checks)
        if described and canonical not in conflicting
    ]
    chosen = sorted(random.Random(PCQM_SAMPLE_SEED).sample(range(len(kept)), PCQM_SAMPLE_SIZE))
    out = dataset_dir / "molecules.csv"
    write_table(out, ["mol_id", "smiles", "gap"], [list(kept[i]) for i in chosen])
    print(f"PCQM4Mv2: {len(rows)} rows, {len(kept)} usable, {len(chosen)} sampled -> {out}")
    return out


def load_molecules(dataset_dir: Path):
    """The molecule table, with property values parsed exactly."""
    import pandas as pd  # noqa: PLC0415

    return pd.read_csv(
        dataset_dir / "molecules.csv",
        dtype={"mol_id": str, "smiles": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
