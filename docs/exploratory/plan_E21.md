# E21: other sampling designs on the 43 fields (written 2026-11-21, before running it)

**Status.** Specified after the study's results had been seen, and after an earlier check of five designs on the 20 factor-setting fields
had been seen (`experiments/sensitivity/designs.py`, kappa = 1.10, `expected/sensitivity/report_designs.txt`); so exploratory. Run once and
reported whatever it shows. The method stays as it is: TabPFN-GP as released (0.0.1), kappa = 1.15, mean correction from 25 samples, and
every comparator with its version and factor of `configs/fixed/calibration.json`. No factor is set again, and the five tests of the study
are not affected.

**Amendment (2026-11-21, before any TabPFN output was computed).** The first lattice rule took the two directions at right angles and
recovered the lattice on 8 of the 26 regular fields: several layouts are staggered or triangular, or have rows farther apart than the
spacing within a row. The rule above replaces it. Check 4 had been computed with the first rule (the sample mean covered more field means
under `grid` than under `gsrs` on those 8 fields) and is computed again.

**Question.** The study revealed simple random samples. On farms, samples are more often taken on a regular grid, a few per management
zone, in a few compact spots, or where a problem is suspected. Do the 80 % intervals of TabPFN-GP and of the comparators still cover about
80 % of field and zone means when the samples are drawn in these ways?

**How the designs are emulated.** No field of the 43 was sampled under several designs (the Midwest release, collected to compare designs,
gives only site, location and coordinates; checked on 2026-11-21). As in the study, each design therefore chooses which of a field's
measured locations are revealed; the others are the targets. The fields, properties, compact zones and targets are those of the study
(means over the unrevealed locations of the field and of each zone with at least 3 unrevealed locations). Each split is seeded by
crc32('field|design|rep'), with the property added for `tsoil`.

**Designs.**
- `grid`: a coarser grid, on the 26 fields with a regular layout (`expected/study/field_areas.csv`). The field's lattice is recovered from
  the coordinates: two lattice directions from the nearest-neighbour vectors, at whatever angle the layout has (square, rectangular,
  staggered or triangular), and each location's index along both; the design takes every second location along both directions, with
  each of its 4 possible starting locations, and every third, with each of its 9, a coarser lattice of the same shape. These 13 samples are all the grid samples at these spacings, so they are taken as they are, without repeats; n is whatever each
  gives (about N/4 and N/9). A sample with fewer than 5 locations is not used.
- `gsrs`: for each field and spacing of `grid`, 10 simple random samples (repeats 400-409) of the median size of that spacing's grid
  samples: the reference for `grid`.
- `strat`: a stratified random sample by the four zones: n/4 locations at random in each zone; where n/4 is not whole (n = 10, 30), two
  zones chosen at random get one location more.
- `clust`: a clustered sample: four random seed locations, then the nearest unrevealed neighbours of each in turn until n are chosen, as
  the cluster designs of `experiments/core/common.py`.
- `tcov`: a sample targeted on a covariate, a model of sampling where an image or map suggests a problem: n locations drawn without
  replacement with probability proportional to exp(1.5 z), where z is the first principal component of the field's standardised
  covariates (as TabPFN receives them, coordinates excluded), scaled to unit variance over the field's locations, with its sign drawn at
  random for each repeat. TabPFN sees these covariates, so it can partly account for the selection.
- `tsoil`: a sample targeted on the property itself, a model of sampling where the crop looks poor: as `tcov`, with z the property's
  values standardised over the field's locations. No method sees what drives this selection.

`strat`, `clust`, `tcov` and `tsoil` use n = 10, 20 and 30, wherever at least 10 locations remain unrevealed, with 10 repeats
(400-409); their reference is the study's own simple random samples at n = 10, 20 and 30 (repeats 300-319, stored rows, no new fits).

**Methods.** Every method of the study except TabPFN-KpR (which needs its own TabPFN fits and kriging, and was used below its intended
sample sizes): TabPFN-GP, the shared-error-only model, TabPFN, block RK, WH and the sample mean, each with its study factor and CRPS
version; the difference estimators of the table of further methods (field residuals for field means, the zone's own residuals for zone
means, factor 1.05); and, for `strat`, the stratified estimator of the field mean (zone weights from the unrevealed locations; t interval
with n - 4 degrees of freedom; the sample mean's factor 1.05 for coverage and no factor for the CRPS, as for the sample mean).

**Code and data.** A new module `experiments/exploratory/designs_study.py` draws the splits and passes them to the study's own functions
(`point_model.our_model` for TabPFN and its cross-fitted residuals, `block_cases.case_rows` and `further_models.case_rows` for the rows,
`scoring` for the scores), so that every method is computed exactly as in the study. No existing module is changed. Inputs: the study's
field files and stored rows. Outputs go to a new folder; no stored file is overwritten. TabPFN runs on the GPU, at low priority; at
most 12 CPU workers for the rows. Estimate: about 10,900 splits
(9,240 for the four designs with fixed n; about 1,700 for `grid` and `gsrs`), about 3.5 h on one GPU and 1-2 h of CPU.

**Check first. If any check fails, stop.**
1. On three fields with different layouts (DK_FAA regular, PO_NE irregular, DR_AUZ clustered), at n = 20, repeats 300 and 301 of the
   study, the new code run on the study's own splits reproduces the stored M and V of TabPFN-GP, the shared-error-only model, TabPFN,
   block RK, WH and the sample mean (relative difference below 1e-6).
2. Every split has n distinct locations and at least 10 unrevealed. `strat` puts its allocation in each zone; `clust` consists of four
   groups grown from their seeds; for `tcov` and `tsoil`, the mean of z at the revealed locations, signed towards the targeted end, is at
   least 0.5 on every field (otherwise the report lists the fields).
3. For `grid`, the lattice is recovered on a field if at least 95 % of its locations lie within a quarter of the spacing of a lattice node
   and no two share a node; fields where it is not are listed and left out of `grid` and `gsrs`. The report lists every field, design and
   size that was skipped, with the reason.
4. A check against theory, before any other result is read: on autocorrelated soil a grid sample estimates the mean more precisely than a
   random sample of the same size, so the sample mean's interval, whose variance assumes random sampling, should cover more field means
   under `grid` than under `gsrs`. If its pooled coverage under `grid` is lower, look for an error before reading anything else.

## Reported

For each design and size, and pooled over the sizes, against its reference on the same fields (`gsrs` for `grid`, the study's samples
for the others):
- coverage of 80 % intervals of field means, zone means and zones without a revealed sample, for every method, with 95 % bootstrap
  intervals over fields, and the change from the reference (paired by field);
- the CRPS ratio of TabPFN-GP to each method and the interval-width ratio, with 95 % bootstrap intervals over fields and the number of
  fields where TabPFN-GP's CRPS was lower;
- coverage by zone sample count (none, one or two, three or more);
- `tcov` and `tsoil` side by side.

The full report goes to `expected/exploratory/report11.txt`, the summary to `report11.csv`.

## Reading (fixed before running)

A method **keeps its coverage** under a design if its pooled coverage lies within [0.75, 0.85], the band of test 3. It **covers too few**
(or **too many**) if the 95 % interval lies wholly below 0.75 (or above 0.85); otherwise its coverage is given as it is.

- **In every case.** One paragraph in the exploratory results and one table in the appendix of results give the coverage of every method
  by design, and the CRPS ratios. The limitation "Only simple random samples were evaluated" is replaced by what was found, the designs
  are described as emulated by subsampling the measured locations, and the repository's README says which designs were tested. The
  method, the factors and the five tests stay as they are.
- **If TabPFN-GP keeps its coverage of field and zone means under `grid` and `strat`:** the caption of the table of estimators says that
  it holds for grid and stratified samples too; the same is said for the sample mean and the difference estimator where it holds for them.
- **If TabPFN-GP covers too few field or zone means under `grid` or `strat`:** the abstract's caveat sentence, the second conclusion and
  the table of estimators say so.
- **If any method covers too few means under `clust`, `tcov` or `tsoil`:** the limitations say that, with clustered or targeted samples,
  the intervals of these methods covered too few means, with the figures given in the results, and the Discussion advises spreading the
  samples over the field.
- **If TabPFN-GP keeps its coverage under a design where the sample mean does not, or the other way round:** the table of estimators
  says so for that design.
