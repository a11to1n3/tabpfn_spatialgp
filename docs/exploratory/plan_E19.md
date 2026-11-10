# E19: the CRPS with each mean weighted by its unrevealed share (written 2026-11-09, before running it)

**Status.** Specified after the study's results had been seen, so exploratory. Run once and reported whatever it shows.

**Question.** The study scores the mean over the unrevealed locations and sums the CRPS over sample sizes. For the mean over all sampled
locations of the same area, the revealed part is known, so the CRPS of that mean is the same CRPS multiplied by the unrevealed share.
Scoring without this factor gives large samples more weight in the sum, most on small fields. Do the CRPS comparisons depend on this?

**The weight.** For a block B, w_B = m_B / (m_B + n_B), the share of the block's sampled locations that is unrevealed: for a field m/N, for
a zone its own share (1 for a zone without a revealed sample). Each block's CRPS is multiplied by w_B for every method. Coverage does not
change, as the interval and the target are scaled together.

**Inputs.** The stored rows of the development splits, the factor-setting data and the 43 fields. CPU only: no new predictions. The
study's scoring, selection and report code is used unchanged; only the CRPS column is multiplied by w_B when an area is scored. Outputs go
to a new folder; no stored file is overwritten.

**Check first. If any check fails, stop.**
1. With every weight equal to 1, the same script gives back the versions and factors of `configs/fixed/` and a report identical
   to the study's.
2. In the rows of the field means, n_B = n and m_B = m; every weight lies in (0, 1].

## Steps

Every choice made with the CRPS is made with the weighted CRPS instead, for the comparators only. TabPFN-GP and its κ = 1.15 stay fixed.
1. **Version.** On the development splits, `experiments/factor_setting/set_factors.py select` chooses each comparator's version with the
   weighted CRPS.
2. **Factor and CRPS version.** On the factor-setting data, `experiments/factor_setting/set_factors.py calibrate`. The factors depend on coverage only, so they differ
   from the study's only for another version; the choice between the version with and without the factor depends on the CRPS.
3. **Evaluation.** On the 43 fields, the study's report (`experiments/study/run.py`) with these versions and factors.

## Reported

For each of the ten contrasts of test 5, unweighted and weighted: the CRPS ratio with its 95 % interval, the number of fields with a lower
CRPS, the Holm-adjusted p-value and the verdict. The CRPS ratio of test 4a and whether the test is met. Each comparator's version, factor and CRPS version. The
share of TabPFN-GP's summed field-mean CRPS from 5 to 20 samples, unweighted and weighted. The full report goes to
`expected/exploratory/report_weighted.txt`, the summary to `report9.txt` and `report9.csv`.

## Reading (fixed before running)

- **In every case.** One clause in the paper's paragraph on aggregation saying that the sums are not weighted by the unrevealed share, and
  one sentence in "Robustness of the tests" giving the result. Tests 4a and 5 as reported are not replaced.
- **If every verdict is unchanged:** nothing else changes.
- **If a verdict weakens from better to no worse:** that sentence names the contrasts and their weighted ratios, and the Results paragraph
  on test 5 points to it.
- **If a contrast is no longer met, or a weighted ratio is above 1:** also stated in the fourth conclusion and in the abstract's sentence
  on the CRPS. Any further change is decided after seeing the result.

## Addition: the study's versions, and the versions with the lowest weighted CRPS (after the result of the steps above had been seen)

**Why.** The steps above use, for each comparator, the simplest version whose weighted CRPS on the development splits is within 0.5 % of
the lowest. For Block RK, the version with the lowest weighted CRPS, `PI_fix`, scores 0.3 % below `RK_ins_fix_v`, the simpler one, and its
factor and CRPS version differ. Two further sets of versions therefore separate the effect of the weights from that of the versions.

**What.** On the 43 fields, with the weighted CRPS:
1. **Weights alone.** The study's versions, factors and CRPS versions (`configs/fixed/`).
2. **Lowest versions.** Each comparator's version with the lowest weighted CRPS on the development splits (`selection_scores.csv` of the
   steps above), its factor and CRPS version set by `experiments/factor_setting/set_factors.py calibrate` with the weighted CRPS.

Stored rows only. Outputs go to a new folder; no stored file is overwritten.

**Check first. If any check fails, stop.**
1. With every weight equal to 1, the same script gives a report identical to the study's.
2. With the weights alone, every line of the report that does not involve the CRPS is identical to the study's, as the weights cannot
   change coverage.
3. With the weights alone, the contrasts of test 5 whose comparator has the same version, factor and CRPS version as in the steps above, and
   the ratio of test 4a, have the same CRPS ratio and number of fields with a lower CRPS as there.
4. With the lowest versions, the same holds for every comparator whose version with the lowest weighted CRPS is the one of the steps above.

**Reported.** For each comparator, its version, factor, CRPS version and weighted development score under 1 and 2. For each of the ten
contrasts of test 5, unweighted, under 1 and under 2: the CRPS ratio with its 95 % interval, the number of fields with a lower CRPS, the
Holm-adjusted p-value and the verdict. Test 4a and the rise in field-mean coverage of test 1 under the same three. The summary goes to
`expected/exploratory/report9b.txt` and `report9b.csv`, the full reports to `expected/exploratory/report_weighted_kept.txt` (1) and
`report_weighted_lowest.txt` (2).

**Reading.** The reading above applies to 1 and 2 in the same way. The sentence in "Robustness of the tests" gives both results and says,
for each contrast that weakens, whether it weakens with the weights alone or only with the lowest versions. The fourth conclusion and the
abstract's sentence on the CRPS, which state any weighted ratio above 1, say the same.
