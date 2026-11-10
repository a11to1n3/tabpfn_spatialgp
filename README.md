# tabpfn_spatialgp

**Field and zone means of a soil property, with honest intervals, from a few laboratory samples.**

TabPFN predicts the property at every location from covariates known everywhere (a conductivity survey, elevation, images). A Gaussian
process (GP) on TabPFN's held-out errors gives the uncertainty of the mean over the field or a zone, including the error shared by the
whole field. The method is called TabPFN-GP. Built with TabPFN.

In the paper's test on 43 fields, with 5 to 100 samples per field, its 80 % intervals covered 82.9 % of field means and 79.4 % of zone
means, and between 79 and 85 % at every sample size.

## Install

```bash
pip install git+https://github.com/a11to1n3/tabpfn_spatialgp.git
```

Python 3.9 or newer; this installs TabPFN 2.0.9, the version used in the paper. A GPU is used if there is one (much faster). The first
fit downloads TabPFN's weights.

## Quickstart

```python
from tabpfn_spatialgp import TabPFNGPRegressor, zone_blocks

# field: one row per location, with covariates and projected coordinates "x", "y" (metres)
# sampled: the rows sent to the laboratory, y_sampled their values; rest: the rows where you want the means
reg = TabPFNGPRegressor(coord_cols=["x", "y"], target_kind="SOC")     # "SOC" (or organic matter), "Clay" or "pH"
reg.fit(field.iloc[sampled], y_sampled, field_coords=field[["x", "y"]].to_numpy())

blocks = zone_blocks(field[["x", "y"]].to_numpy(), rest)              # the whole field and four zones
means = reg.predict_blocks(field.iloc[rest], blocks)
means["field"]   # {'mean': ..., 'sd': ..., 'interval80': (low, high), ...}
```

`python examples/quickstart.py` runs this on a synthetic field. The estimator also has `predict`, `predict_quantiles`,
`predict_distribution` and `report`, and works with `cross_val_score`, `Pipeline` and `GridSearchCV`.

**Good to know**
- Pass `field_coords`, the coordinates of all sampled locations of the field: they set the field's size, which scales the GP's prior.
- Coordinates must be in metres; use `crs="lonlat"` for degrees. Missing covariate values are fine, missing coordinates are not.
- Memory grows with the square of the number of prediction rows: predict at the sampled locations or on a coarse grid.
- Tested on means over sampled locations, 5 to 100 random samples, compact zones and TabPFN 2.0.9. Coverage varied between fields
  (0.77 to 0.89 for field means, 0.72 to 0.88 for zone means); zones without samples and large zones were covered less often. Dense
  grids, other sampling designs and zones defined by covariates were not tested.

## Structure

```
src/tabpfn_spatialgp/   the package; every setting is in settings.py
examples/               a runnable example
tests/                  tests: pytest tests; the GPU tests run with RUN_GPU=1
configs/                the GP's prior, and the comparators' variants and every factor that the study reads (fixed/)
docs/                   the study's plan (PROTOCOL.md) and the plans of the development checks and exploratory analyses
experiments/            the code behind the paper, one folder per stage (experiments/README.md)
  core/                 shared modules: the sets of splits, TabPFN, the block model, the comparators, the scores
  data/                 the development and factor-setting field files
  pipeline/             point outputs, and the field and zone means of every method
  development/          TabPFN-GP's settings and the GP's prior, on the development fields
  factor_setting/       the comparators' variants and every factor
  study/                the study on 43 fields: downloads, data rules, predictions, report
  sensitivity/          the sensitivity checks in the paper's settings table
  exploratory/          the exploratory analyses (experiments/exploratory/README.md)
paper/                  the scripts that make the paper's tables and figures from the outputs
expected/               the outputs behind the paper
environment/            the exact Python and R versions
reproduce.sh            runs every stage, from the public data to the study's report
```

Every module of `experiments/` runs from the repository root, as `python -m experiments.<stage>.<module>`.

## Reproducing the paper

You need Linux with an NVIDIA GPU (CUDA 12.4; the study used one RTX 3090), Python 3.11, R 4.2.2 with GDAL, GEOS and PROJ, about 10 GB
of disk and an internet connection.

```bash
git clone https://github.com/a11to1n3/tabpfn_spatialgp.git && cd tabpfn_spatialgp
python3.11 -m venv .venv
.venv/bin/pip install -r environment/requirements.txt && .venv/bin/pip install -e .
Rscript environment/install_r_packages.R ~/R/soilgp && export SOILGP_RLIB=~/R/soilgp
PY=.venv/bin/python bash reproduce.sh all
```

This takes about 8 hours and writes everything to `work/` (set `WORK=...` for another folder). Stages can be run one at a time, in this
order, and a stage that stops resumes when run again:

| Stage | What it does | Time |
|---|---|---|
| `data` | Builds the development and factor-setting field files | 1 min |
| `dev` | TabPFN and TabPFN-KpR on the development splits (GPU) | 21 min |
| `prior` | Derives the GP's prior | 1 min |
| `factor_setting` | Chooses the comparators' variants and sets every factor, κ included (GPU); checks them against `configs/fixed/` | 2 h 10 min |
| `study` | The study on 43 fields (GPU): downloads, data rules, predictions, report and figure | about 5.5 h |
| `compare` | Compares the outputs with `expected/` | seconds |

- The five tests and the other results of the study are in `work/study/report/report.txt`.
- The exploratory analyses run after the study; their commands are in `experiments/exploratory/README.md`.
- On another GPU or CUDA version, a few numbers can move in the third decimal. The Sentinel-2 scene of a field can differ with the network
  state; `export_log.csv` names the scene used.
- TabPFN runs on one BLAS thread with a fixed seed, because for some inputs its output depends on the thread count.

## Data

All data are open. LimeSoDa (CC BY-SA 4.0), the Lithuanian SensRes fields (CC BY 4.0) and Oxford and Meuse (GPL) are the development
and factor-setting data. The sources and licences of the 43 fields of the study are in Annex A of `docs/PROTOCOL.md`; their checksums are
in `experiments/study/fetch.sh` and `experiments/study/sources_further.sha256`.

## Citation and licence

Please cite the paper, *Estimating field and zone means of soil properties from sparse samples: A case study combining TabPFN and
Gaussian process regression* ([`CITATION.cff`](CITATION.cff)), TabPFN (Hollmann et al., 2025, *Nature* 637, 319–326) and the data you
use. The code is MIT-licensed; TabPFN and its weights are under TabPFN's own licence.

For questions or problems, please open an issue.
