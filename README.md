# Diversity sampling costs labels near the training pool and saves them far from it

Code for the AI4Mat-NeurIPS 2026 workshop paper. It reproduces the experiments
from the public QM9 and PCQM4Mv2 files up to the two quantities the paper reports
per distance decile:

- **r = MAE<sub>MaxMin</sub>(N) / MAE<sub>Random</sub>(N)**, the error ratio of
  diversity (MaxMin) sampling to random sampling at the same number of labels N;
- **M/N**, where M is the number of random labels that reaches the error of
  MaxMin with N labels.

Distance is the mean Tanimoto distance (ECFP4, 2048 bits) from a test molecule
to its 100 nearest molecules in the train pool U.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/). All package versions are
pinned in `uv.lock`.

```bash
uv sync
```

The MLP and Chemprop runs use a GPU when one is available (`torch` 2.13 with CUDA 13).

## 1. Data

```bash
uv run python scripts/prepare_data.py qm9
uv run python scripts/prepare_data.py pcqm --workers 32
```

This downloads the raw files into `data/<dataset>/raw/`, then writes

| path | content |
|---|---|
| `molecules.csv` | cleaned molecules; every index below is a row of this table |
| `features/ecfp.npy`, `features/rdkit2d.npy` | ECFP4 bits, 217 RDKit 2D descriptors |
| `splits/<split>/{train_pool,iid_test,ood_test}.npy` | U, IID test and OOD test |
| `splits/<split>/{iid,ood}_test_distance.npy` | distance of each test molecule to U |

Cleaning and splits:

- **QM9** (130,831 molecules): SMILES from the official XYZ files, properties from
  MoleculeNet's `gdb9.sdf.csv`; molecules listed in `uncharacterized.txt` and
  molecules RDKit cannot parse are removed. Targets: HOMO-LUMO gap, α, μ.
- **PCQM4Mv2**: molecules with a gap value that RDKit parses and describes are kept,
  canonical SMILES carrying conflicting gap values are removed, and 1.5M molecules
  are sampled (seed 42).
- `property_<target>`: OOD = the 10,000 molecules with the lowest kernel density of
  the target value; IID test = 10% of the rest (QM9) or 10,000 of the rest (PCQM).
  For PCQM, `property_gap_150k` keeps the same test sets with a 150k subsample of U.
- `scaffold` (QM9): among molecules with a ring scaffold, 20% are drawn as test;
  test molecules whose Murcko scaffold occurs in U are IID, the others OOD.
- `cluster` (QM9): ECFP → PCA(64) → k-means (k=10); cluster 0 is OOD, and an IID
  test set of the mean cluster size is drawn from the other clusters.

If a host refuses the scripted download, download the file by hand and put it in
`data/<dataset>/raw/` (the URLs are in `src/divsample/data.py`); its hash is checked.

**Checking the data.** At the end, the script compares hashes of every array with
`checksums/<dataset>.json`, the data used in the paper. With the locked package
versions all of them should match. The k-means clustering is the step most
likely to differ on other hardware (floating-point order in BLAS); if only
`splits/cluster/*` differs, the cluster split is a different but equally valid draw.

## 2. Sweep

```bash
uv run python scripts/run_sweep.py configs/qm9_gap.yaml
uv run python scripts/run_sweep.py configs/qm9_alpha.yaml
uv run python scripts/run_sweep.py configs/qm9_mu.yaml
uv run python scripts/run_sweep.py configs/pcqm_gap.yaml
```

For every split, condition (model and input), N ∈ {100, 300, 1000, 3000, 10000}
and seed 0–4, the sweep selects N molecules from U by `random` and by `maxmin`
(RDKit `MaxMinPicker` on ECFP4 Tanimoto distance, from a random first pick),
fits the model on them and predicts the IID and OOD test sets.

| model | inputs in the paper |
|---|---|
| kernel ridge (RBF, targets centered on the selected labels) | ECFP, RDKit 2D |
| random forest (500 trees) | ECFP, RDKit 2D |
| MLP (64-32, 1000 epochs) | ECFP, RDKit 2D |
| Chemprop 2 (50 epochs) | SMILES |

QM9 gap uses all seven conditions; QM9 α and μ and PCQM gap use Chemprop, KRR on
RDKit 2D and MLP on ECFP. RDKit 2D inputs are standardized on the selected
molecules. Hyperparameters are in `src/divsample/models.py` and can be overridden
under `params:` in a config.

Results go to `results/<dataset>_<target>/<split>/<model>_<input>/n<N>/seed<s>/<strategy>/`
(selected indices and test predictions). Finished runs are skipped, so a sweep can
be restarted. `--shard i/n` runs every n-th run, to spread a sweep over n jobs.

## 3. r and M/N

```bash
uv run python scripts/compute_metrics.py configs/qm9_gap.yaml          # N = 1000
```

writes `results/<dataset>_<target>/metrics_n1000_cells.csv` (one row per split,
test set, condition, seed and decile) and `metrics_n1000_summary.csv` (one row per
split, test set and decile). For each condition and seed, M is read off that seed's
random learning curve by interpolating log MAE against log N, after replacing the
curve by its running minimum; M beyond the sampled sizes is censored at 100 or
10,000 and counted in `n_seeds_censored_low/high`. The summary averages log r and
log₂ M/N over conditions within a seed, then gives the mean and SD across seeds.

## Reproducibility

Selections, splits, distances, KRR and random forest predictions are deterministic
given the locked versions. KRR results can change in the last digits with the number
of BLAS threads. The MLP sets `torch.use_deterministic_algorithms(True)`, but GPU
and CPU runs differ; Chemprop runs are not bitwise reproducible.

## Layout

```
configs/          one sweep per dataset and target
scripts/          prepare_data.py, run_sweep.py, compute_metrics.py
src/divsample/    data, features, splits, distance, selection, models, sweep, metrics
checksums/        hashes of the data used in the paper
tests/
```

## Citation

```bibtex
@inproceedings{diversity-sampling-2026,
  title     = {Diversity Sampling Costs Labels Near the Training Pool and Saves Them Far From It},
  booktitle = {AI4Mat Workshop at NeurIPS 2026},
  year      = {2026},
}
```

## License

MIT
