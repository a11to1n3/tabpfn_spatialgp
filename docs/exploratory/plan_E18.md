# E18: the comparators' variogram settings without the fields with unreliable coordinates (written 2026-11-09, before running it)

**Status.** Specified after the study's results had been seen, so exploratory. Run once and reported whatever it shows.

**Question.** Block RK and WH take the range and the nugget share of their variogram or correlogram from medians of fits on the 14
development fields. SL.125 and UL.120 are released with dummy coordinates, and SP.231 spans several fields. TabPFN-GP's prior has this
check (the row "Prior refitted without SL.125, UL.120" of the table of further methods); the comparators' settings do not. Do Block RK's
and WH's results, and the tests that involve them, depend on these fields?

**Two sets of fields for the medians.**
- X2: the 12 development fields without SL.125 and UL.120, as in the check of TabPFN-GP's prior.
- X3: the 11 development fields without SL.125, UL.120 and SP.231.

**Inputs.** The stored fits (`work/factor_setting/shape_fits.csv`), the stored point outputs of the development splits, the factor-setting data and
the 43 fields, and the stored rows. CPU and R only: no new TabPFN predictions. Outputs go to a new folder; no stored file is overwritten.
TabPFN-GP, TabPFN, TabPFN-KpR, the sample mean and the shared-error-only model do not use these settings, so their rows are reused.
Each set is recomputed with the code and settings of its stored rows.

**Check first. If any check fails, stop.**
1. The medians over all 14 fields from `shape_fits.csv` equal `configs/fixed/shapes.json`, and the medians of X2 and X3 are
   (range/extent, nugget share): Block RK (`r_ins`) X2 0.075, 0.108; X3 0.077, 0.091. WH (`z_oof`) X2 0.069, 0.151; X3 0.074, 0.029.
2. With the unchanged settings, the new code reproduces the stored rows of every variant that uses the settings (`M` and `V` within
   1e-9) on one field of each of the three sets.
3. From the stored rows, the study's own code gives back the versions `RK_ins_fix_v` and `WH_fix`, the factors 1.55 and 1.60, and the
   study's report of the five tests.

## Steps, for each of X2 and X3

Every choice these settings inform is made with the settings of the set, with the study's code unchanged.
1. **Settings.** Medians of range/extent and nugget share per residual type (`r_ins`, `r_oof`, `z_oof`) over the fits of the set's fields.
2. **Version.** On the development splits (all 14 fields, as stored), recompute the variants that use the settings (`RK_ins_fix`,
   `RK_ins_fix_v`, `RK_oof_fix`, `RK_oof_fix_v`, `PI_fix`, `WH_fix`); the other variants are unchanged. `experiments/factor_setting/set_factors.py select` then chooses
   each comparator's version.
3. **Factor.** On the factor-setting data (all 20 fields, as stored, the three fields included, as for TabPFN-GP's check), the same
   variants; `experiments/factor_setting/set_factors.py calibrate` sets the factor and the CRPS version, with the comparators' weighting of the study.
4. **Evaluation.** On the 43 fields, the same variants; the study's report (`experiments/study/run.py`) with the versions and factors of steps 2 and 3,
   on the areas scored by all methods.

Not recomputed: the other analyses that use Block RK and WH (one factor rule, factors for each sample size, seeds, means over more locations).

## Reported

For Block RK and WH, for X2 and X3: the medians; the version chosen; the factor; the CRPS version; coverage of field and zone means; the
rise in coverage D of field and zone means; the CRPS ratio TabPFN-GP / comparator for field and zone means, with its 95 % interval and
the verdict of test 5. Also the outcome of test 1 over all four comparators and of test 5 over all ten contrasts. The full report of each
set goes to `expected/exploratory/report_shapes_X2.txt` and `report_shapes_X3.txt`, the summary to `report8.txt` and `report8.csv`.

## Reading (fixed before running)

- **In every case.** One sentence in the appendix with the details of the methods: all 14 fields set the medians, with their values
  without these fields. Four rows in the table of further methods (Block RK and WH, X2 and X3), marked as having the factors and CRPS
  versions of the main comparison, so that they compare directly with it. Table 3 and the five tests as reported are not replaced.
- **If test 1 is still met and every test 5 verdict for Block RK and WH is unchanged, for both X2 and X3:** nothing else changes.
- **If a test 5 verdict weakens from better to no worse:** also one sentence in "Robustness of the tests".
- **If test 1 or a test 5 contrast is no longer met:** stated in "Robustness of the tests", and the Discussion's sentence that a
  variogram form taken from other fields did not by itself stop the coverage from rising is revised to match. Any further change is
  decided after seeing the result.
