# E23: zones formed from the soil property on the 43 fields (written 2026-11-22, before running it)

**Status.** Specified after the study's results and after the analyses of sampling designs (E21) and of zones from covariates (E22)
had been seen. In E22, TabPFN-GP covered 0.775 of covariate-zone means, against 0.794 of compact-zone means, but those zones separated
the properties less than the compact zones did (median share of a property's variance between zones 0.100 against 0.145; note added after the run: these are medians of the shares
printed to two decimals in `report12.txt`, and over the exact shares, in `report13.txt`, they are 0.104 and 0.142), so they did
not test zones with sharp soil boundaries. No earlier check of such zones exists. The zones below are built to be hard for (A3), so the
result may be unfavourable. The zone rule was fixed before any run, without trying another rule. Exploratory; run once and reported
whatever it shows. The method stays as it is: TabPFN-GP as released (0.0.1), kappa = 1.15, mean correction from 25 samples, and every
comparator with its version and factor of `configs/fixed/calibration.json`. No factor is set again, and the five tests of the study
are not affected.

**Question.** On farms, zones for variable-rate lime or fertiliser are meant to separate soils whose property levels differ, so their
boundaries cross sharp changes in the property. Where a zone's soil differs from the rest of the field, TabPFN's errors may differ
systematically between zones (for example, too low in a high-pH zone and too high in a low-pH zone), which TabPFN-GP's single error
process for the whole field (A3) does not describe. Do the 80 % intervals of TabPFN-GP and of the comparators still cover about 80 % of
zone means when the zones separate the property sharply?

**Zones.** No field of the 43 comes with delineated management zones or soil-map units, so the zones are formed from the measured
property itself, for each field and property, at all N locations of the field:
1. each location gets the mean of the property at its four nearest other locations (by the coordinates; on a square grid, the four
   locations next to it). Its own value is left out, so a location's zone is not chosen by its own measurement error, and the
   averaging over neighbours makes the zones contiguous patches rather than scattered single locations;
2. these neighbour means are clustered by k-means into 4 zones (random seed 0, best of ten starts, as the compact zones).

The zones are bands of the smoothed property, with boundaries where the property changes. They use the measured values at locations
that are later targets, which no farm can do, so they separate the property more sharply than zones a farm delineates from covariates,
yield maps or the grower's judgement: a hard case, not a farm design. The zones differ between the properties of a field (pH zones for
lime, SOC or clay zones otherwise). As in the study, a zone is evaluated where it holds at least 3 unrevealed locations, and its target
is the mean over its unrevealed locations. No method receives the property values used to form the zones; the methods receive only the
zone's locations, as for the compact zones, and the zone-effect model the zone labels, as a farm's zone map would give them.

**Splits and outputs.** Every split of the study (all sample sizes, repeats 300-319), with the study's stored point outputs of TabPFN,
the cross-fitted residuals and TabPFN-KpR; no TabPFN fit is repeated. Only the zones change; field means are the study's and are not
reported again.

**Methods.** As in E22: every method of the study, each with its study factor and CRPS version (TabPFN-GP, the shared-error-only
model, TabPFN, TabPFN-KpR, block RK, WH and the sample mean of the zone's revealed values); the within-zone difference estimator
(factor 1.05); and the zone-effect model with independent effects of these zones (its factor 1.20). The zone-effect model is the
evaluated method that does describe errors that differ between zones, where the zone holds revealed samples.

**Code and data.** A new module `experiments/exploratory/property_zones.py` uses the functions of E22's
`experiments/exploratory/covariate_zones.py` (replacing the zones of `block_cases.blocks_of` and the labels of the zone-effect model),
with the zones keyed by field and property instead of by field, and otherwise calls the study's own functions, so every method is
computed exactly as in the study. No existing module is changed. Inputs: the study's field files, stored point outputs and stored rows.
Outputs go to a new folder; no stored file is overwritten. CPU only, at most 12 workers at low priority; estimate about
30-45 min (E22 took 27 min).

**Check first. If check 1, 3 or 4 fails, stop.**
1. With the compact zones left in place, the new code reproduces the stored rows of every method (as in E22's check 1: DK_FAA, PO_NE,
   DR_AUZ, n = 20, repeats 300 and 301, relative difference below 1e-6).
2. For every field and property: the sizes of the four zones; their agreement with the compact and with the covariate zones (adjusted
   Rand index); the share of the property's variance between zones for property, covariate and compact zones; and how contiguous the
   zones are (share of locations whose nearest other location lies in the same zone), for the three zone sets. Zones never evaluated
   are listed with the reason.
3. A check against theory, before any other result is read: the zones are fixed before sampling, so under simple random sampling the
   sample mean of a zone's revealed values stays unbiased for the zone's mean; if its pooled coverage lies below 0.75, look for an error
   before reading anything else.
4. The zones must answer the question: if the median share of a property's variance between property zones is not above that of the
   compact zones (0.145), they are not sharp, and only check 2 is reported.

## Reported

As in E22, for the property zones and, on the same splits, the compact zones of the study:
- coverage of 80 % intervals of zone means for every method, with 95 % bootstrap intervals over fields, and the change from the compact
  zones (paired by field); also by the zone's number of revealed samples (none, one or two, three or more), by property, by group of
  fields, for fields with released and with public covariates only, by sample size, and by how sharply the zones separate the property
  (field-property pairs in thirds of the between-zone share);
- the rise in coverage D for zone means (n >= 30 minus n <= 10) for every method, with 90 % and 95 % intervals;
- the CRPS ratio of TabPFN-GP to each method and the interval-width ratio, with 95 % bootstrap intervals over fields and the number of
  fields where TabPFN-GP's CRPS was lower;
- the descriptions of check 2.

The full report goes to `expected/exploratory/report13.txt`, the summary to `report13.csv`.

## Reading (fixed before running)

A method **keeps its coverage** of property-zone means if its pooled coverage lies within [0.75, 0.85], the band of test 3. It **covers
too few** (or **too many**) if the 95 % interval lies wholly below 0.75 (or above 0.85); otherwise its coverage is given as it is.

- **In every case.** The exploratory paragraph on zones formed from covariates gets the result for zones formed from the property, and
  the table of zone results in the appendix gets a second panel with the same columns (its generator extended). The zones are described
  as formed from the measured property, sharper than zones a farm can delineate, not as management zones. The limitation's clause on
  errors that differ between zones (A3) is replaced by what was found. The repository's README says what was tested. The method, the
  factors and the five tests stay as they are.
- **If TabPFN-GP keeps its coverage of property-zone means:** the table of estimators says, for its zone rows, that it held for zones
  that separate the property sharply too (exploratory).
- **If TabPFN-GP covers too few property-zone means:** the abstract's caveat sentence, the second conclusion and the table of estimators
  say so; the Discussion relates it to (A3), one error process for the whole field, and, where another evaluated method kept its
  coverage of these zone means, names it in the table of estimators for zones that separate soils sharply.
- **If TabPFN-GP's coverage is given as it is:** the table of estimators gives it for these zones, without a verdict.
- **In zones without a revealed sample:** if TabPFN-GP covered fewer property-zone means than compact-zone means (95 % interval of the
  paired change wholly below 0), the Discussion's advice to flag zones without samples says that it matters more where zones separate
  soils sharply.
- **CRPS:** reported as found; the abstract and conclusions change only through the coverage readings above.
