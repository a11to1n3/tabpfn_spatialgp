# Exploratory analysis E6: a fix for the interval tails (written 2026-10-26, before evaluating it on the 43 fields)

## Why

E4 showed TabPFN-GP's intervals:
- slightly too wide at 50 %;
- too narrow at 90 and 95 % for zone means, and at 95 % for field means with n <= 10.

The 80 % intervals, which the study tested, are fine.

## Diagnosis on the factor-setting data only

Script `experiments/exploratory/tails_diagnosis.py`, output `diag_tails_first.txt`. Data: first-set rows, each external task once, as for kappa = 1.15.

- The same pattern appears in the factor-setting data.
- The ratio of the 95 % to the 80 % quantile of |obs - M| / V^(1/2) is 2.1 at n = 5, against 1.53 for a normal distribution. It falls to about 1.5 to 1.7 from n = 30 on.
- Zones without a revealed sample add little.

Cause: TabPFN-GP plugs in its total variance (assumption A4), which is estimated from n held-out errors. With few errors, this estimate is uncertain, and that gives heavier tails than the normal distribution. The usual remedy, as for the sample mean, is a Student-t distribution with n - 1 degrees of freedom.

## Candidates compared on the factor-setting data only

Script `experiments/exploratory/tails_fixes.py`, output `diag_fixes_first.txt`.

- **C1: one factor per level,** by the kappa rule.
- **C2: Student-t shape with nu = n - 1,** scaled so that the 80 % interval is unchanged. Half-width at level a, with p = (1 + a) / 2: kappa * z_0.9 * V^(1/2) * t_nu(p) / t_nu(0.9).
- **C3: C2 plus a factor per level.**

**Chosen: C2.**
- Nothing is fitted, and every 80 % result of the study stays exactly as it is.
- On the factor-setting data, 95 % coverage rose:
  - field means from 0.939 to 0.950 (n <= 10: from 0.908 to 0.929);
  - zone means from 0.932 to 0.940.
- 50 % coverage of field means fell from 0.563 to 0.551.
- C1 over-covered at 95 % with n >= 30 (0.969).
- C3's three extra factors added little at 95 %. They pushed 50 % coverage of zone means below 0.50 (0.487).

## Evaluation of C2 on the 43 fields

This is exploratory: it was specified after the study's results had been seen. It is run once, on the area sets of the report, and reported whatever it shows. Script `experiments/exploratory/student_t.py`.

1. **Coverage at 50, 80, 90 and 95 %:**
   - for n <= 10, n >= 30 and all n;
   - for field and zone means;
   - 95 % bootstrap intervals over fields (10,000 resamples, seed 0).

   Check: the 80 % coverage must equal TabPFN-GP's.
2. **Rise in coverage D at 50 and 95 %,** for TabPFN-GP and for C2, with the study's `rise` (fields with both bands).
3. **CRPS** (location-scale t CRPS, as for the sample mean), with the study's aggregation:
   - C2 against TabPFN-GP;
   - C2 against each comparator.
4. **Width of 95 % intervals,** C2 against TabPFN-GP.

## Reporting

- Appendix C gets a short table (coverage at the four levels, D at 95 %, CRPS ratio).
- The Results get one or two sentences.
- The Limitations say that the fix was set on the factor-setting data, specified after the results were seen, and evaluated on the same 43 fields, so it needs new fields to confirm it.
- The method stays as tested. The t shape is described as an option.
