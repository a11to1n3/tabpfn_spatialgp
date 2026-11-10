# expected

The outputs behind the paper, one folder per stage of `experiments/`.

| Folder | Outputs | Written by |
|---|---|---|
| `development/` | `report_development.txt`, `report_dev_nugget.txt`, `report_dev_mu2.txt` | `experiments/development/` |
| `factor_setting/` | `report_first.txt` | `experiments/factor_setting/analysis.py` |
| `study/` | `report.txt`, `coverage_by_n.csv`, `export_log.csv`, `field_areas.csv` | `experiments/study/` |
| `sensitivity/` | `report_sensitivity.txt`, `report_designs.txt`, `timing.csv`, `timing_cpu.csv` | `experiments/sensitivity/` |
| `exploratory/` | The outputs of the exploratory analyses | `experiments/exploratory/` (its README lists them) |

`bash reproduce.sh compare` compares a new run, line by line, with `factor_setting/report_first.txt`, `study/export_log.csv`,
`study/report.txt` and `study/coverage_by_n.csv`, and checks that the prior derived in stage `prior` is that of `configs/priors.json`
and of the package. The other files are the outputs of analyses that `reproduce.sh` does not run.
