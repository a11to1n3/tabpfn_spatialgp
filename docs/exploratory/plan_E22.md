# E22: zones from covariates on the 43 fields (written 2026-11-21, before running it)

**Status.** Specified after the study's results had been seen, after the design analysis (E21) and after an earlier check of zones from
covariates on 14 factor-setting fields had been seen (`experiments/sensitivity/sensitivity.py`, zone set `c4`, kappa = 1.10,
`expected/sensitivity/report_sensitivity.txt`, section 4: TabPFN-GP covered 0.731 of these zone means with n <= 20 and 0.755 with
n >= 30, the sample mean 0.807 and 0.816, and TabPFN-GP 0.613 in zones without a revealed sample with n <= 20); so exploratory, and the earlier check
suggests the result may be unfavourable. Run once and reported whatever it shows. The method stays as it is: TabPFN-GP as released
(0.0.1), kappa = 1.15, mean correction from 25 samples, and every comparator with its version and factor of
`configs/fixed/calibration.json`. No factor is set again, and the five tests of the study are not affected.

**Question.** The study's zones are compact clusters of the coordinates. On farms, management zones separate soils for variable-rate
lime or fertiliser and are delineated from covariates such as conductivity surveys, elevation, imagery or yield maps, so their
boundaries follow the soil pattern. TabPFN's errors may then differ systematically between zones, which TabPFN-GP's single error
process for the whole field (A3) does not describe. Do the 80 % intervals of TabPFN-GP and of the comparators still cover about 80 % of
zone means when the zones are formed from the covariates?

**Zones.** The field files hold locations, covariates and properties only; no field of the 43 comes with delineated management zones.
Zones are therefore formed from each field's covariates, as the earlier check formed them: k-means with 4 clusters (random seed 0, best
of ten starts, as the compact zones) of the principal components that carry 90 % of the variance of the field's standardised covariates
(as TabPFN receives them, coordinates excluded), at all N locations of the field. A covariate with a missing value at any location is
left out of the clustering; a field with no covariate left is listed and left out. The clusters need not be contiguous. They stand in
for management zones: no yield maps, smoothing or a farmer's judgement are used. As in the study, a zone is evaluated where it holds at
least 3 unrevealed locations, and its target is the mean over its unrevealed locations.

**Splits and outputs.** Every split of the study (all sample sizes, repeats 300-319), with the study's stored point outputs of TabPFN,
the cross-fitted residuals and TabPFN-KpR; no TabPFN fit is repeated. Only the zones change; field means are the study's and are not
reported again.

**Methods.** Every method of the study, each with its study factor and CRPS version: TabPFN-GP, the shared-error-only model, TabPFN,
TabPFN-KpR, block RK, WH and the sample mean of the zone's revealed values; the within-zone difference estimator of the table of further
methods (factor 1.05); and the zone-effect model of that table with independent effects of the four covariate zones instead of the
compact ones (its factor 1.20, set by the one rule for every factor).

**Code and data.** A new module `experiments/exploratory/covariate_zones.py` replaces the zones of `block_cases.blocks_of` (and the
zone labels of the zone-effect model) by the covariate zones and otherwise calls the study's own functions (`block_cases.load_case`,
`block_cases.case_rows`, `further_models.case_rows`, `scoring`), so every method is computed exactly as in the study. No existing module
is changed. Inputs: the study's field files, stored point outputs and stored rows. Outputs go to a new folder; no
stored file is overwritten. CPU only, at most 12 workers at low priority; estimate about 1-2 h.

**Check first. If check 1 or 3 fails, stop.**
1. With the compact zones left in place, the new code reproduces the stored rows of every method of the study (M and V of TabPFN-GP, the
   shared-error-only model, TabPFN, TabPFN-KpR, block RK, WH and the sample mean) on three fields with different layouts (DK_FAA
   regular, PO_NE irregular, DR_AUZ clustered), at n = 20, repeats 300 and 301 (relative difference below 1e-6).
2. For every field: the number of covariates and components used, the sizes of the four covariate zones, the agreement of the covariate
   zones with the compact zones (adjusted Rand index), and for each property the share of its variance between zones under each zone
   set. Fields left out, and zones never evaluated, are listed with the reason.
3. A check against theory, before any other result is read: under simple random sampling the sample mean of a zone's revealed values is
   unbiased for the zone's mean whatever the zone's shape, so its coverage of covariate-zone means should stay near the stated level, as
   for the compact zones. If its pooled coverage lies below 0.75, look for an error before reading anything else.

## Reported

For the covariate zones and, on the same splits, the compact zones of the study:
- coverage of 80 % intervals of zone means for every method, with 95 % bootstrap intervals over fields, and the change from the compact
  zones (paired by field); also by the zone's number of revealed samples (none, one or two, three or more), by group of fields, for
  fields with released and with public covariates only, and by sample size;
- the rise in coverage D for zone means (n >= 30 minus n <= 10) for every method, with 90 % and 95 % intervals;
- the CRPS ratio of TabPFN-GP to each method and the interval-width ratio, with 95 % bootstrap intervals over fields and the number of
  fields where TabPFN-GP's CRPS was lower;
- the descriptions of check 2.

The full report goes to `expected/exploratory/report12.txt`, the summary to `report12.csv`.

## Reading (fixed before running)

A method **keeps its coverage** of covariate-zone means if its pooled coverage lies within [0.75, 0.85], the band of test 3. It **covers
too few** (or **too many**) if the 95 % interval lies wholly below 0.75 (or above 0.85); otherwise its coverage is given as it is.

- **In every case.** One paragraph in the exploratory results and one table in the appendix of results give the coverage of every method
  for covariate and compact zones, the change, and the CRPS ratios. The limitation that management zones were not evaluated is replaced
  by what was found, with the zones described as formed from covariates, not delineated on the farm. The repository's README says what
  was tested. The method, the factors and the five tests stay as they are.
- **If TabPFN-GP keeps its coverage of covariate-zone means:** the table of estimators says, for its zone rows, that it held for zones
  from covariates too (exploratory).
- **If TabPFN-GP covers too few covariate-zone means:** the abstract's caveat sentence, the second conclusion and the table of estimators
  say so; the Discussion relates it to (A3), one error process for the whole field, and, where another evaluated method kept its
  coverage of these zone means, names it in the table of estimators for zones from covariates.
- **If TabPFN-GP's coverage is given as it is:** the table of estimators gives it for zones from covariates, without a verdict.
- **In zones without a revealed sample:** if TabPFN-GP covered fewer covariate-zone means than compact-zone means (95 % interval of the
  paired change wholly below 0), the Discussion's advice to flag zones without samples says that it matters more for zones from
  covariates.
- **CRPS:** reported as found; the abstract and conclusions change only through the coverage readings above.
