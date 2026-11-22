# experiments/exploratory

The exploratory analyses in the paper's Results and in its appendix *Results by group, field and sample size*. They were specified after
the study's results had been seen; each plan (`docs/exploratory/plan_E*.md`) was written before its analyses were run, and every result
is reported. They use the study's code unchanged and κ = 1.15 unless stated. Their outputs are in `expected/exploratory/`; `paper/` makes
the paper's tables from them, run in the paper's `figs/source` folder.

Run the commands from the repository root, with `W` the folder written by `reproduce.sh` (default `work/`), `ST=$W/study` and `OUT` any
output folder. GPU analyses need the TabPFN environment; analyses that krige need R with `gstat` and `automap`.

| | Analysis | Plan | Output (`expected/exploratory/`) | In the paper |
|---|---|---|---|---|
| E1, E2 | CRPS and width ratios by sample size; comparators' factors tuned for each sample size on these results | `plan_E1-E2.md` | `explore_report.txt`, `explore_by_n.csv` | The figure of the CRPS ratio by sample size; `tab_explore` |
| E3 | Rise in coverage by group of fields and by property | `plan_E3-E5.md` | `explore2_report.txt`, `explore2.csv` | `tab_bygroup_rise` |
| E4 | Coverage of 50, 90 and 95 % intervals | `plan_E3-E5.md` | As E3 | `tab_levels` |
| E5 | Computing time of every method (GPU, run alone) | `plan_E3-E5.md` | `timing_methods.csv` | `tab_timing` |
| E6 | Student-t shape with n − 1 degrees of freedom; diagnosis and choice on the factor-setting data | `plan_E6.md` | `diag_tails_first.txt`, `diag_fixes_first.txt`, `explore3_report.txt`, `explore3.csv` | `tab_tshape` |
| E7 | The five tests with κ = 1.10 | `plan_E7-E8.md` | `report_k110.txt` | `tab_robust` |
| E8 | TabPFN-GP with seeds 0–4 on ten fields (GPU) | `plan_E7-E8.md` | `seeds_report.txt` | Not reported; E12 repeats it on all 43 fields |
| E9–E11, E13 | Inference by dataset and group; one factor rule for all methods; further models and components; comparators with a factor for each sample size | `plan_E9-E13.md` | `report5.txt`, `report5.csv`, `factors_once.json`, `report_once.txt`, `report_once_gp1.txt` | `tab_robust`, `tab_further`, the last block of `tab_explore`; the exploratory section of the Results |
| E12 | All methods with seeds 1–4 of TabPFN and of the fold split (GPU, about 2.5 h per seed); the measures of E8 on all 43 fields | `plan_E9-E13.md` | `report_seed<s>.txt`, `seeds_spread.txt` | `tab_robust`; the settings table |
| E14 | Means over nested subsets of the unrevealed locations of each field and zone, and the dense limit | `plan_E14.md` | `dense_report.txt` | `tab_dense`; the exploratory section of the Results |
| E15–E17 | TabPFN-GP without its factor; zones without a revealed sample, further models; zones by number of revealed samples | `plan_E15-E17.md` | `report7.txt`, `report7.csv` | `tab_further`; the exploratory section of the Results |
| E18 | Block RK's and WH's variogram settings from fewer development fields | `plan_E18.md` | `report8.txt`, `report8.csv`, `report_shapes_X2.txt`, `report_shapes_X3.txt` | `tab_further`; appendix with the details of the methods |
| E19 | The CRPS weighted by the unrevealed share: the comparators' versions set with it, the study's versions, and the versions with the lowest weighted CRPS | `plan_E19.md` | `report9.txt`, `report9.csv`, `report9b.txt`, `report9b.csv`, `report_weighted.txt`, `report_weighted_kept.txt`, `report_weighted_lowest.txt` | "Robustness of the tests"; test 5 in the Results; abstract; fourth conclusion |
| E20 | What is subtracted from TabPFN's variance | `plan_E20.md` | `report10.txt`, `report10.csv` | table of settings; predictive distribution; (A4) |
| E21 | Other sampling designs, chosen among the measured locations: grid, stratified, clustered and targeted samples (GPU) | `plan_E21.md` | `report11.txt`, `report11.csv` | `tab_designs`; the exploratory section of the Results; limitations; table of estimators |
| E22 | Zones formed from covariates (k-means of their principal components) on the study's splits (CPU and R) | `plan_E22.md` | `report12.txt`, `report12.csv` | `tab_covzones`; the exploratory section of the Results; limitations; table of estimators |
| E23 | Zones formed from the soil property (k-means of each location's mean of the property at its four nearest other locations) on the study's splits (CPU and R) | `plan_E23.md` | `report13.txt`, `report13.csv` | `tab_covzones` (panel b); the exploratory section of the Results; Discussion; limitations; table of estimators; abstract; conclusions |

## Commands

```bash
# E1, E2
python -m experiments.exploratory.by_sample_size $ST/rows OUT

# E3, E4
python -m experiments.exploratory.groups_and_levels $ST/rows OUT

# E5 (GPU)
SOILGP_DATA=$W/data SOILGP_RUN=$W/run python -m experiments.exploratory.timing OUT

# E6
python -m experiments.exploratory.tails_diagnosis $W/factor_setting
python -m experiments.exploratory.tails_fixes $W/factor_setting
python -m experiments.exploratory.student_t $ST/rows OUT

# E7
SOILGP_DATA=$ST/data python -m experiments.exploratory.kappa_110 $ST/rows OUT

# E8 (GPU)
SOILGP_DATA=$ST/data python -m experiments.exploratory.seeds_ten_fields OUT
python -m experiments.exploratory.seeds_ten_fields_report OUT $ST/rows OUT

# E9-E11, E13: rows of the further models on the factor-setting data and on the 43 fields, the field areas, then the report
SOILGP_DATA=$W/data SOILGP_RUN=$W/run python -m experiments.exploratory.further_models_rows first $ST/variants 8
SOILGP_DATA=$ST/data SOILGP_RUN=$ST/run python -m experiments.exploratory.further_models_rows study $ST/variants 8
python -m experiments.study.field_areas $ST/csv > $ST/report/field_areas.csv
python -m experiments.exploratory.further_models_report $ST/rows $ST/variants $W/factor_setting $ST/report/field_areas.csv OUT

# E12 (GPU, about 2.5 h per seed)
WORK=$ST PY=python JOBS=8 bash experiments/exploratory/run_seeds.sh
python -m experiments.exploratory.seeds_spread $ST OUT

# E14
SOILGP_DATA=$ST/data SOILGP_RUN=$ST/run python -m experiments.exploratory.dense_limit rows OUT 12
python -m experiments.exploratory.dense_limit report OUT $ST/rows

# E15-E17
python -m experiments.exploratory.factor_and_zones $ST/rows $ST/variants expected/exploratory/factors_once.json OUT

# E18 (reads factor_setting/ and run/points_dev, run/points_first of $W)
REPRO=$W W=$ST PY=python bash experiments/exploratory/run_variogram_settings.sh

# E19 (OUT2: the folder of the second report)
SOILGP_DATA=$ST/data SOILGP_RUN=$ST/run python -m experiments.exploratory.weighted_crps $W/factor_setting $ST/rows $ST/report/report.txt OUT
SOILGP_DATA=$ST/data SOILGP_RUN=$ST/run python -m experiments.exploratory.weighted_crps_versions $W/factor_setting $ST/rows $ST/report/report.txt OUT OUT2

# E20 (STORED: the rows of the other alternatives of the table of settings, for the check)
SOILGP_DATA=$W/data SOILGP_RUN=$W/run python -m experiments.development.dev_nugget rows $STORED 8
SOILGP_DATA=$W/data SOILGP_RUN=$W/run python -m experiments.exploratory.variance_subtraction rows OUT 8
SOILGP_DATA=$W/data SOILGP_RUN=$W/run python -m experiments.exploratory.variance_subtraction report OUT $STORED

# E21 (GPU for check and points, about 15 h as one job; splits and checks first, then TabPFN, the rows and the report)
export SOILGP_DATA=$ST/data SOILGP_RUN=OUT/run
python -m experiments.exploratory.designs_study splits --out OUT
python -m experiments.exploratory.designs_study check --out OUT --study-rows $ST/rows
python -m experiments.exploratory.designs_study points --out OUT
python -m experiments.exploratory.designs_study rows --out OUT --workers 12
python -m experiments.exploratory.designs_study report --out OUT --study-rows $ST/rows

# E22 (CPU and R, about 30 min with 12 workers; zones and check 2, check 1, then the rows and the report, which starts with check 3)
export SOILGP_DATA=$ST/data SOILGP_RUN=$ST/run
python -m experiments.exploratory.covariate_zones zones --out OUT
python -m experiments.exploratory.covariate_zones check --out OUT --study-rows $ST/rows --variants $ST/variants
python -m experiments.exploratory.covariate_zones rows --out OUT --workers 12
python -m experiments.exploratory.covariate_zones report --out OUT --study-rows $ST/rows --variants $ST/variants

# E23 (CPU and R, about 30 min with 12 workers; zones with checks 2 and 4, check 1, then the rows and the report, which starts with check 3)
python -m experiments.exploratory.property_zones zones --out OUT
python -m experiments.exploratory.property_zones check --out OUT --study-rows $ST/rows --variants $ST/variants
python -m experiments.exploratory.property_zones rows --out OUT --workers 12
python -m experiments.exploratory.property_zones report --out OUT --study-rows $ST/rows --variants $ST/variants
```

- `report_once.txt` and `report_once_gp1.txt` are `OUT/report_once/report.txt` and `OUT/report_once_gp1/report.txt`.
- E12 (`run_seeds.sh`, which calls `seed_study.py`) writes `$ST/run_seed<s>`, `$ST/rows_seed<s>` and `$ST/report_seed<s>/report.txt`.
  Seed 0 is the study itself.

The sensitivity checks in the paper's settings table are `folds.py`, `sensitivity.py`, `designs.py` and `models_report.py` in
`experiments/sensitivity/`, and `dev_nugget.py` and `dev_mu2.py` in `experiments/development/`; their outputs are in
`expected/sensitivity/` and `expected/development/`.
