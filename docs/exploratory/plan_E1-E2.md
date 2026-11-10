# Exploratory analyses of the study run (written 2026-10-20, before running them)

These analyses were specified after the results of the study (report.txt) had been seen. They are exploratory: no decision rules, no
p-values. They use only the saved area rows of the study run (`work/rows`), the study's methods, and the area sets and aggregation of
`experiments/study/run.py` (scoring.score, scoring.contrast). Each is run once, and every result is reported, whatever it shows.

## E1. CRPS and interval width by sample size

For each sample size n in {5, 10, 15, 20, 30, 50, 75, 100}, each kind of area (field, zone) and each comparator (TabPFN, TabPFN-KpR,
block RK, WH, sample mean):

- the CRPS ratio TabPFN-GP / comparator, aggregated as in test 5 but restricted to the areas with that n: per task the log ratio of mean
  CRPS, mean over the field's properties, geometric mean over fields; 95 % percentile interval from 10,000 resamples of fields (seed 0);
  number of fields where TabPFN-GP is lower;
- the 80 % interval-width ratio, aggregated the same way.

The study's factors and CRPS versions are used, on the same area sets as the report: the four TabPFN-based comparators on the areas
where all five TabPFN-based methods are scored; the sample mean on its own areas with TabPFN-GP.

Check before reporting: aggregated over all n, the CRPS and width ratios must equal report.txt (test 5 and the width line).

## E2. Comparators with factors tuned for each sample size (best case for the comparators)

TabPFN-GP keeps its single factor, kappa = 1.15. Each comparator instead receives a separate factor for each sample size and each kind
of area. The factor is chosen on these evaluation results themselves, which favours the comparators.

- **CRPS factor.** phi on the grid 0.05, 0.10, ..., 20.00 minimising the comparator's CRPS aggregate at that n and kind: the mean over
  fields of the mean over properties of log(mean CRPS), on the areas of E1. This equals maximising the CRPS ratio TabPFN-GP /
  comparator at that n. For the sample mean, the scale of its t distribution is multiplied by phi.
- **Coverage factor.** The smallest grid value at which the comparator's pooled 80 % coverage at that n and kind reaches 0.80.

Reported for each n, kind and comparator:
- the CRPS factor and the CRPS ratio TabPFN-GP / comparator at that factor, with a 95 % interval and the number of fields lower;
- the coverage factor, the comparator's coverage at it, TabPFN-GP's coverage, and the width ratio TabPFN-GP / comparator at that factor.

Also reported, over all n: the CRPS ratio with the comparators at their CRPS factors, and the width ratio with them at their coverage
factors.

The intervals hold the tuned factors fixed: the factors are not re-tuned within the bootstrap.

## Presentation (no new analysis)

From report.txt only:
- a scorecard table in the Results;
- a figure of the CRPS ratios by group of fields;
- a figure of coverage by field.

## Reporting

The paper states that these analyses were specified after the results had been seen and are exploratory. Results go in Appendix C, with a short summary in the Results.
