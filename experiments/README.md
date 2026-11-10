# experiments

The code behind the paper, one folder per stage. `reproduce.sh` runs the stages in order. Every module runs from the repository root, as
`python -m experiments.<stage>.<module>`; its docstring gives the arguments.

| Folder | What it holds | Stage of `reproduce.sh` |
|---|---|---|
| `core/` | Modules shared by every stage | |
| `data/` | The development and factor-setting field files | `data` |
| `pipeline/` | Point outputs, and the field and zone means of every method | `dev`, `factor_setting` |
| `development/` | TabPFN-GP's settings and the GP's prior, on the 14 development fields | `prior` |
| `factor_setting/` | The comparators' variants and every factor | `factor_setting` |
| `study/` | The study on 43 fields | `study` |
| `sensitivity/` | The sensitivity checks in the paper's settings table | |
| `exploratory/` | The exploratory analyses (`exploratory/README.md`) | |

`compare.py` compares a reproduction with `../configs/fixed/` and `../expected/` (stages `factor_setting` and `compare`); `paths.py`
holds the folders of the repository.

## core

- `datasets.py` -> the fields, properties and designs, and the data folder.
- `config.py` -> every fixed choice: the sets of splits (`dev`, `first`, `final`), the candidate variants, the factor grid and thresholds.
- `common.py` -> TabPFN, training sets, predictive distributions, the R bridge and output files.
- `point_model.py`, `kriging.R` -> the point outputs of one split: TabPFN with cross-fitted errors, and TabPFN-KpR with its kriging
  features.
- `block_model.py` -> TabPFN-GP's block model: the distribution of field and zone means.
- `block_cases.py`, `block_kriging.R` -> the field and zone means of every method for one split.
- `scoring.py`, `aggregate.py` -> the scores (CRPS, coverage, width), their aggregation and the tests.

## data

- `export_data.py`, `export_sensres.R`, `export_external.R` -> the field files `<field>.npz`: covariates `X`, properties `Y`, coordinates
  in metres and LimeSoDa's folds. SL.125 and UL.120 have dummy coordinates, given for privacy.
- `checks.py` -> checks of the field files and of the package.

## pipeline

- `points.py` -> the point outputs of a set of splits (GPU).
- `block_rows.py` -> the comparators' variogram settings, and the field and zone means of every method on a set (CPU and R).
- `further_models.py` -> the shared-error-only model and the difference estimators.

## development

- `derive_priors.py` -> the GP's prior from the 630 development fits (`../configs/priors.json`, `../configs/prior_fits.csv`).
- `development.py`, `ablation.py`, `gp_variants.py` -> TabPFN-GP's development choices (`../docs/development/plan_development.md`);
  output `../expected/development/report_development.txt`.
- `dev_nugget.py`, `dev_mu2.py` -> two further checks of the block model (`../docs/development/plan_dev_checks.md`); outputs
  `../expected/development/report_dev_nugget.txt` and `report_dev_mu2.txt`.

## factor_setting

- `set_factors.py` -> the choice of the comparators' variants on the development splits and of every factor on the factor-setting data;
  its outputs, which the study reads, are in `../configs/fixed/`.
- `analysis.py` -> the scores and tests of a set of area rows; output `../expected/factor_setting/report_first.txt`.

## study

`../docs/PROTOCOL.md` is the study's plan and `run_study.sh` runs it (`reproduce.sh study` calls it); its outputs are in
`../expected/study/`.

```bash
WORK=/path/to/work PY=/path/to/python SOILGP_RLIB=/path/to/Rlib JOBS=6 bash experiments/study/run_study.sh all
```

The stages are `fetch`, `export`, `data`, `points` (GPU), `rows` and `report`; each can be run alone. The report is
`$WORK/report/report.txt`.

- `fetch.sh`, `sources_further.sha256` -> downloads every source and checks its checksum.
- `export.R` -> the locations, properties and covariates of every field, by the data rules of the plan.
- `run.py` -> the field files, predictions, area rows and report.
- `field_areas.py` -> the area and layout of each field (Table 1 of the paper).

## sensitivity

Run after the study; outputs in `../expected/sensitivity/`.

- `folds.py` -> TabPFN's cross-fitted errors with other numbers of folds, and computing time (`timing.csv`, `timing_cpu.csv`).
- `sensitivity.py` -> TabPFN-GP with one fixed choice changed at a time (`report_sensitivity.txt`).
- `designs.py` -> other sampling designs and seeds (`report_designs.txt`).
- `models_report.py` -> other values of κ, further models and coverage by property.

## Folders

`SOILGP_DATA` (field files; default `work/data`), `SOILGP_RUN` (run outputs; default `work/run`) and `SOILGP_RLIB` (R library, if
needed).
