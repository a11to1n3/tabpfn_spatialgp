# Protocol: TabPFN-GP intervals for field and compact-zone soil means

This protocol fixes the whole study before any part of it runs: methods, data, data rules, design, endpoints, secondary analyses and
reporting. The study is run once. Every endpoint and secondary analysis below is reported whatever its outcome, and no other analysis
enters the paper.

## 1. Aim

For means of a soil property over the remaining sampled locations of a field and of compact zones, estimated from a simple random
sample of n = 5 to 100 locations, evaluate the coverage and the continuous ranked probability score (CRPS) of 80 % intervals of
TabPFN-GP against TabPFN-based comparators and the sample mean, and test whether intervals without a term for the error shared by the
field are too narrow with few samples and too wide with many.

## 2. Methods (all fixed; nothing is tuned in this study)

- **TabPFN-GP**: package `tabpfn_spatialgp` 0.0.1 (tag `v0.0.1`): kappa = 1.15, correction of TabPFN's mean from n = 25, the
  population prior and averaging grid of `configs/priors.json`, Matern-3/2 correlation, five-fold cross-fitting; TabPFN 2.0.9
  with its released weights and the package's settings.
- **Shared-error-only model**: TabPFN-GP without the spatial part (sigma^2 = 0), kappa = 1.15.
- **Comparators**, with the variants, factors and CRPS versions of `configs/fixed/selection.json` and `calibration.json`:
  TabPFN with independent errors (factor 2.40); TabPFN-KpR (factor 4.20; CRPS without the factor); block regression kriging of
  TabPFN's in-sample errors with the cross-field variogram plus TabPFN's variance (factor 1.55); the error-correlogram method of
  Wadoux and Heuvelink with the cross-field correlogram (factor 1.60); the sample mean (t interval, factor 1.05).
- These were all fixed on the 14 development fields and on the 20 LimeSoDa fields with Oxford and Meuse. None of the fields of this
  study informed any of them.

## 3. Data

### 3.1 Fields
All fields of the five groups in Annex A that meet the rules of Section 3.3. Raw files are downloaded from the original sources and
checked against the SHA-256 checksums of Annex A before any property value is read.

### 3.2 Properties
Soil organic carbon (or organic carbon, organic matter or, where none of these is released, total carbon), clay and pH of the topsoil
layer each release reports first, as defined in Annex A, each in the release's units.

### 3.3 Data rules (one script, every field)
1. A location without coordinates, more than 5 km from its field's median coordinates, or sharing coordinates with another location is
   excluded.
2. Plausibility: a pH outside 2-11, an organic or total carbon content outside 0-60 % (0-600 g/kg), an organic matter content outside
   0-100 %, or a clay content outside 0-100 % (0-1000 g/kg) is treated as missing.
3. A record whose determinations (four or more) all share one value is treated as corrupted; all its values are missing.
4. Repeated analyses of one sample: identical repeats are used once; for particle size, the analysis whose five fractions sum to
   100 % (1000 g/kg) is used; other differing repeats are averaged.
5. A field's locations are those with a value of each of its properties. A property missing at more than 10 % of a field's locations
   is not used for that field. A field needs at least 30 locations.

### 3.4 Covariates (one rule)
The covariates supplied with a field (Annex A) that are numeric and known at every location: a point value, or a raster or grid read
at the cell or record nearest the location, known there if that record lies within 10 m. Otherwise elevation and slope from the
Copernicus DEM GLO-30 (reprojected to the field's coordinate system at 30 m) and the ten bands of one bare-soil Sentinel-2 L2A scene
(Earth Search; April, May, October or November 2019-2023; cloud cover at most 10 %; every location's scene classification 4 or 5; the
lowest median NDVI over the locations; ties to the earliest). A covariate without a value, or constant, at the field's locations is
dropped; a field left without covariates is not evaluated. No covariate derived from a soil property is used.

## 4. Design

- Sample sizes n in {5, 10, 15, 20, 30, 50, 75, 100}, each used where at least 10 locations remain; 20 repeats per sample size
  (repeats 300-319, seeded by field, sample size and repeat as in the package's evaluation code).
- Areas: the field and four k-means clusters of the coordinates (seed 0, ten starts); a zone is scored if at least three of its
  locations remain. Targets: means over the remaining locations.
- Areas scored by all methods: those where TabPFN-GP and the four TabPFN-based comparators all return a prediction; the sample mean
  on the areas where it is defined.

## 5. Endpoints

Fields are the unit; intervals are percentile intervals from 10,000 bootstrap resamples of the fields (seed 0), the areas of each
resampled field pooled. Coverage is that of 80 % intervals over the areas scored by all methods. The rise in coverage D of a method is
its coverage over the areas with n in {30, 50, 75, 100} minus that with n in {5, 10}, over the fields with both groups.

- **H1, coverage rises with n without a shared-error term.** For each of TabPFN, TabPFN-KpR, block RK and WH, field means: met if the
  95 % lower bound of D is above 0 for all four.
- **H2, TabPFN-GP's coverage does not rise.** TabPFN-GP's D for field means and for zone means separately: met if the 90 % interval
  lies within [-0.10, 0.10].
- **H3, calibration.** TabPFN-GP's coverage of field means and of zone means separately: met if the 95 % interval lies within
  [0.75, 0.85].
- **H4, variance terms.** On all areas where both are scored: (a) the shared-error-only model's field-mean coverage interval lies
  within [0.75, 0.85] and its CRPS ratio to TabPFN-GP has an upper bound below 1.02; (b) TabPFN-GP's zone coverage minus the
  shared-error-only model's has a 95 % lower bound above 0.
- **H5, CRPS.** The CRPS ratio of TabPFN-GP to each comparator (TabPFN, TabPFN-KpR, block RK, WH, the sample mean), for field and zone
  means: a field's value is the mean over its properties of the log ratio of summed CRPS over its areas; TabPFN-GP is better if the
  one-sided exact Wilcoxon signed-rank test over fields, Holm-corrected over the 10 comparisons, gives p < 0.05 with a ratio below 1;
  otherwise it is no worse if the upper 95 % bound is below 1.02.

Each endpoint has its own rule; there is no overall verdict.

## 6. Secondary analyses (reported with intervals, no pass/fail)

Coverage by n for every method; D for zone means of the comparators and for the sample mean; coverage and CRPS ratios by group of
fields; interval-width ratios of TabPFN-GP to each method; coverage in zones without a sample; coverage by field and by property;
counts of failed predictions per method and sample size.

## 7. Reporting and code

- Failed predictions are counted and reported, never replaced.
- One pipeline (`experiments/study/run_study.sh`): fetch and check sources, export, covariates, predictions, area rows, report. Before the run, a
  dry run on one development field (BB.72, two repeats) checks the code only; its outputs are not reported.
- The paper reports Sections 5 and 6 and nothing else.

## Annex A. Datasets

| Group | Source, licence | Fields | Properties and layer | Coordinates | Supplied covariates |
|---|---|---|---|---|---|
| Uniformity trials | geoR 1.9-6 (CRAN, GPL), data soil250 and soja98 | SOIL250, SOJA98 | SOIL250: C (SOC), Argila (clay), pHAgua; SOJA98: MO (organic matter), PH; topsoil as released | grid row and column (SOIL250); X, Y (SOJA98) | Cota (SOIL250); PROD, iCone (SOJA98) |
| SensRes | Zenodo 10.5281/zenodo.14161809 (CC BY 4.0): SensRes_Denmark.zip, SensRes_Northern_Ireland.zip, SensRes_Turkey.zip | Estrup, Faardrup, Jyndevad, Silstrup, Vindum (Denmark); Brecart A, Maryland (Northern Ireland); Cihanbeyli, Kadinhani, Karapinar, Karatay (Turkey) | Denmark: humus (organic matter), clay; Northern Ireland: SoilTC (total carbon), Clay, SoilpH; Turkey: soc, clay, ph | as released (UTM) | every raster of the field's sensor folders (EMI, gamma, satellite, drone) |
| Midwest | Potash et al. 2023, figshare files 41539671 (locations.csv) and 42223749 (measurements.csv), CC BY 4.0 | NE, IL-RS, IL-PT, IL-DG, IL-MC, IL-BR, IL-RD, IL-RT | SOC concentration (SOCc) averaged over 0-30 cm, thickness-weighted, where segments cover exactly 0-30 cm | WGS84 to UTM 14N (NE), 15N (IL-BR), 16N (others) | none |
| New fields | Laamrani et al., Zenodo 6611475; STEROPES Zenodo 14134737 and 14135622; GSIF 0.5-5.1 (CRAN), data cookfarm; Agroscope Zenodo 14076347 (CC BY 4.0, GPL) | Ontario; Klucov, Nova Ves, Prestavlky, Udrnice; Centro Natta, Grandi, Pedrazzini; R.J. Cook Agronomy Farm; HB, ZO | Ontario: SOC (%); Czech sites: SOC, Clay (0-20 cm); Italian fields: CO (5-10 cm); Cook farm: PHIHOX of the horizon starting at 0 cm; Swiss fields: C_org, clay, pH_H2O (0-20 cm) | as in study/export.R | Ontario: every supplied covariate column; Cook farm: DEM, TWI, NDRE.M, NDRE.Sd, Cook_fall_ECa, Cook_spr_ECa (10 m grid) |
| Further fields | INRAE CAREX (Recherche Data Gouv 10.15454/SDUUSO, 10.57745/2V46MF, 10.57745/HRCQL8, 10.57745/WVMIEX; Etalab 2.0); USDA-ARS cottondata (commit 856b570, CC0); Ag Data Commons 1408098 (public domain); EIDC f3118fa8 (OGL); Ag Data Commons 29922563 (CC0) | Auzeville, Epoisses, Ploudaniel, Crouel; F013B4, F033, F111; Walnut Creek; Henfaes, North Wyke; Caroline County field 1 | as in study/export.R (CAREX first horizon: c_org, Argile 0-2 um, ph_eau; Epoisses without SOC; Maricopa SLCLY_M 0-30 cm; Walnut Creek ZCLAY; EIDC Organic_Matter and Soil_pH; Caroline Clay_pct and pH at 0-5 cm) | as in study/export.R | resistivity grids (CAREX), EM38 surveys with defined reading columns (Maricopa F033), terrain metrics (Walnut Creek, Caroline), site characteristics (EIDC) |

Checksums: the SHA-256 values in `experiments/study/fetch.sh` (geoR, SensRes, Midwest, new fields) and `experiments/study/sources_further.sha256`.
