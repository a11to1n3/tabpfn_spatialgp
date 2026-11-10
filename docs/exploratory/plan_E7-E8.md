# Robustness checks E7 and E8 (written 2026-10-26, before running them)

Both checks are specified after the study's results had been seen, and are reported whatever they show.

## E7. The five tests with κ = 1.10

The comparators' factors were set with the factor-setting data weighted as stored (Oxford twice, Meuse's organic matter three times). The same rule with that weighting gives κ = 1.10 for TabPFN-GP. The study used κ = 1.15, the rule with each task counted once.

E7 recomputes the report from the saved area rows of the study (`work/rows`) with κ = 1.10 for TabPFN-GP and the shared-error-only model:
- every other setting and every comparator factor unchanged;
- `experiments/study/run.py`'s own report code, with its KAPPA set to 1.10;
- output `work/report_k110/report.txt`.

Reported: the five tests and their verdicts. Only κ changes, so the comparators' numbers must equal `report.txt` (check).

## E8. Random seeds of TabPFN and of the fold split, on study fields

**Fields.** The first two fields of each group, in the analysis plan's order: SOIL250, SOJA98, DK_EST, DK_FAA, PO_NE, PO_RS, NX_ON, NX_CZK, DR_AUZ, DR_EPO.

**Splits.** The study's splits with n in {5, 20, 50} where defined, repeats 300–304, every property of the field.

**Model.** For each split, TabPFN's outputs are recomputed with seeds 0–4 (the seed of TabPFN and of the fold split), using `designs.tabpfn_outputs`, and TabPFN-GP's field and zone rows are rebuilt with `designs.block_rows` (the study's block model, population prior).

**Check.** Seed 0 must reproduce the study's TabPFN-GP rows (M and V; relative difference below 1e-6, or report the largest difference).

**Reported, with κ = 1.15:**
- the standard deviation over the five seeds of each interval endpoint, divided by the half-width (median and 90th percentile), for field and zone means;
- coverage of 80 % intervals for each seed (field and zone means), and its range over the seeds;
- the CRPS of each seed relative to seed 0 (pooled ratio of summed CRPS over the areas).
