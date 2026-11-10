# E20: what is subtracted from TabPFN's variance (written 2026-11-09, before running it)

**Status.** Specified after the study's results had been seen, so exploratory. Run once and reported whatever it shows. The method stays as
it is: nothing here changes TabPFN-GP or its results on the 43 fields.

**Question.** The independent part of TabPFN-GP's variance at a target location is max{0, v_T(s_a) - v_bar}. The term v_bar is the GP's own
share of TabPFN's variance: the GP's leave-one-out latent variance, averaged over the revealed locations, with the mean beta treated as
known. Would another form of that share give better field and zone means?

**Four versions.** The GP fit, its block moments and the mean are the same in all four; only the amount subtracted differs.
- `base`: the method, subtracting v_bar.
- `marginal`: subtracting sd(r)^2 sigma_bar^2, with sigma_bar^2 the spatial variance mixed over the averaging grid: the GP's share before it
  has seen any neighbour.
- `loo_unknown`: subtracting the same average as v_bar with beta unknown, sd(r)^2 sum_theta omega_theta (1/n) sum_i max{0, 1/[P_theta]_ii -
  tau_theta^2}, where P_theta = K^-1 - K^-1 1 (1' K^-1 1)^-1 1' K^-1.
- `local`: subtracting at each target location its own posterior latent variance, sd(r)^2 sum_theta omega_theta [Sigma_theta]_aa, with
  Sigma_theta the posterior covariance of the latent error at the target locations, including the uncertainty of beta, as in the block
  variance.

In all four, negative remainders are truncated at zero, as in the method.

**Inputs.** The development splits of the 14 development fields, from the stored point outputs, as for the other alternatives of the table of
settings (GP's nugget, all of v_T as independent): CPU only, no new TabPFN predictions, the code those alternatives were run with. Outputs
go to a new folder; no stored file is overwritten.

**Check first. If any check fails, stop.**
1. `base` reproduces the stored rows of those alternatives on all 14 fields (`M` and `V` within 1e-9): 6954 blocks per version, coverage
   of field and zone means 0.780 and 0.765 at kappa = 1, and 0.827 and 0.807 at kappa = 1.10.
2. On one split of every field, for every grid point: the new v_bar equals the code's; 1/[P_theta]_ii equals the leave-one-out variance
   computed directly, by dropping location i and predicting it with beta estimated from the rest (relative difference below 1e-6);
   1/[P_theta]_ii is at least 1/[K^-1]_ii; the diagonal of Sigma_theta equals the code's point variance at the target locations, and the
   mean of Sigma_theta over a block equals the code's block variance for that grid point (relative difference below 1e-8).

## Reported

For `marginal`, `loo_unknown` and `local`, each against `base`, for field and zone means:
- at kappa = 1 and kappa = 1.10, as for the other alternatives: coverage, and the CRPS ratio with its 95 % bootstrap interval over fields
  and the number of fields with a lower CRPS;
- with each version's own factor (see below): the factor, coverage, and the same CRPS ratio;
- coverage by sample size (5 to 10, 15 to 20, 30 or more) and, for zones, by number of locations (3 to 9, 10 to 29, 30 or more), for each
  of the three factors;
- how much is subtracted, as a ratio to v_bar: median and quartiles over the cases, by sample size.

The full report goes to `expected/exploratory/report10.txt`, the summary to `report10.csv`.

**Own factor.** The three alternatives subtract at least as much as `base` at most locations, so at a fixed factor they lower coverage. Since
the coverage of `base` at kappa = 1 is already below 0.80, a comparison at a fixed factor would favour `base` by construction. Each version
therefore also gets its own factor: the smallest on the grid of the comparators' rule (0.50 to 5.00 in steps of 0.05) for which the mean of
field and zone coverage reaches 0.80, set on these same splits for all four versions. The readings below use this comparison.

## Reading (fixed before running)

A version is **clearly better** (or **clearly worse**) than `base` for a kind of mean if its CRPS ratio with its own factor is below 0.995
(or above 1.005) and the 95 % interval excludes 1. The margin is the 0.5 % of the comparators' choice of version.

- **In every case.** TabPFN-GP is unchanged, and the 43 fields are not used unless that is decided after seeing the result. The row on
  the independent part of the table of settings gives the CRPS ratios and the change in coverage of `marginal`, `loo_unknown` and `local`,
  with the kappa values of the other alternatives and with each version's own
  factor. Assumption (A4) points to that row.
- **If no version is clearly better or clearly worse, for field or zone means:** one clause in the section on the predictive distribution
  says that the form of the subtracted share made little difference.
- **If `base` is clearly better than a version:** that clause says so and names the version.
- **If a version is clearly better for field or zone means:** the row and that clause say so, and one sentence in the limitations states
  that TabPFN-GP keeps the form chosen before the evaluation and that this version gave a lower CRPS on the development fields.
- **If, with its own factor, a version's coverage of field means with 30 or more samples minus that with 5 to 10 samples differs from
  that of `base` by more than 0.05:** the row says so, whichever way it goes.
