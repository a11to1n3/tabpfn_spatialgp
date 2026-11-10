# Exploratory analyses E3–E5 (written 2026-10-26, before running them)

They cover three further questions:
- E3: replication across groups of fields and soil properties;
- E4: other interval levels;
- E5: the computing cost of every method.

They were specified after the study's results had been seen and are exploratory: no decision rules, no p-values. Each is run once and reported whatever it shows.

E3 and E4 use only the saved area rows of the study run (`work/rows`), the study's methods and factors, and the helpers of `experiments/study/run.py`. E5 is a timing measurement on the study's GPU machine.

## E3. Replication by group of fields and by soil property

For every method (TabPFN-GP, the shared-error-only model, TabPFN, TabPFN-KpR, block RK, WH, sample mean), for field and zone means:
- the rise in coverage D (n in 30–100 minus n in 5–10) within each of the five groups of fields, and within each property (SOC, clay, pH);
- 95 % percentile intervals from 10,000 bootstrap resamples of that group's fields (seed 0), using the area sets of the report.

A group or property enters only for fields that have both sample-size bands. The uniformity trials have two fields, so their intervals are reported but are not informative.

## E4. Coverage at other interval levels

For every method, with its single factor for all sample sizes (TabPFN-GP: kappa = 1.15), the coverage of central 50, 90 and 95 % intervals of field and zone means:
- pooled over all sample sizes;
- with n <= 10;
- with n >= 30.

Intervals:
- normal predictives: M ± z_q · phi · V^(1/2);
- the sample mean: its t quantile with n_B − 1 degrees of freedom.

Each estimate gets a 95 % bootstrap interval over fields, on the area sets of the report. The 80 % row must reproduce `report.txt` as a check.

## E5. Cost of every method on the same splits

**Splits.** The splits of the existing `timing.csv`: the 20 fields of the factor-setting data without Oxford and Meuse, repeat 120, every sample size, SOC.

**Hardware.** TabPFN on one RTX 3090 GPU; everything else on one CPU thread, as in the study.

**What is timed.** Each method's whole computation for one split, from the revealed data to the field and zone means:
- **TabPFN:** one fit and prediction at all locations, then the area sums.
- **TabPFN-GP:** the same, plus the five fold fits, the GP and the area moments.
- **WH:** TabPFN, the five fold fits, then the correlogram and the area quadratic forms. The correlogram is computed in R (gstat), as in the study.
- **Block RK:** TabPFN with in-sample predictions at the revealed locations, then block kriging in R (gstat).
- **TabPFN-KpR:** the leave-one-out kriging features in R (automap), then one TabPFN fit without coordinates.
- **Sample mean:** the area means and variances.

**R start-up.** R start-up is part of the R methods as implemented, so it is timed separately (`Rscript` loading gstat) and reported alongside.

**Reported.** Median time per split by sample size and the ratio to TabPFN alone. The model loading and the first CUDA call are excluded, as in `timing.csv`.

## Reporting

- Results go in Appendix C.
- The Results get a short summary, and the Discussion's cost paragraph is updated with E5.
- The analysis-plan statement in the paper lists these analyses as exploratory, specified after the results had been seen.
