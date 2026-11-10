# Plan: TabPFN-GP's development choices on the 14 development fields

The comparisons and rules below were fixed, and committed to the repository, before any of the computations was run.

Five choices of TabPFN-GP are set on development splits of the 14 development fields:
1. TabPFN's pipeline: four ensemble members and no target transform.
2. Released rather than fine-tuned weights.
3. TabPFN version 2.0.9 rather than version 3.5.
4. Averaging the GP over the 7 x 5 grid.
5. The block model and the mean correction from n = 50.

Each choice keeps the comparison and, where one existed, the decision rule of the method's development. Where none existed, the rule
below is used.

A choice is kept if its rule selects the current value; otherwise it changes. Every outcome is reported.

## Data and splits

- Fields: BB.250, BB.72, G.150, MG.112, B.204, G.104, MGS.101, SP.231, SL.125, UL.120, H.138, SC.93, LYD.112 and RUM.70.
- The splits are those of the development comparisons, seeded by crc32('field|design|rep'):
  - choices 1, 2, 4 and 5: random 5–90 % designs, replicates 15–19;
  - choice 3: random 5–90 % and cluster 20/40 % designs, replicates 25–29.
- The 12 LimeSoDa fields use the stored outputs of these comparisons, read-only; LYD.112 and RUM.70 are computed with the same code
  and settings.
- Every point GP uses the point prior (`configs/priors.json`, `point`) and every block GP the population prior (`block`).

## Choices, comparisons and decision rules

**1. Pipeline (ablation block A; rule Q1).** On the small band (5–30 %), unit field, pooled SSE ratio with a
field-cluster bootstrap, two-sided Wilcoxon and Holm within the block:
- the pipeline (xy4 against TabPFN's default, xy with eight members and its target transform) is kept if the ratio is below 1 with Holm
  p < 0.05;
- four members are kept unless eight are better (ratio of four over eight above 1 with Holm p < 0.05).

**2. Weights (ablation block C; rule Q2).** Fine-tuned weights replace the released ones only if fine-tuning on real data, alone or
with synthetic GP data, gives a ratio below 1 with Holm p < 0.05 and is seed-robust (every single-seed pair below 1).
- The fine-tuned checkpoints of the development are used. Their training pool holds no development field: neither LYD.112 nor RUM.70, which is
  checked first.

**3. Version (version 3.5).** Version 3.5 replaces 2.0.9 only if its backbone (four members, no target transform),
against 2.0.9's backbone with the same pipeline, both without the GP, gives an SSE ratio below 1 with a 95 % upper bound below 1 on the
random 5–90 % designs.
- The comparison is of backbones only, without the GP, so it does not depend on a GP prior.

**4. Averaging (the screen of GP variants and the ablation's averaging contrast).**
- On the chosen backbone's stored cross-fitted residuals, every GP variant of the screen is refitted with the point prior:
  G0, Ghet, Gdrift, Gtwo, Gall, Gbma, Ghetbma, Gbma0, Gbmashr, Gbma0shr and G0shr.
- As in the screen, the best variant has the lowest mean log SSE ratio against TabPFN alone at each of 10, 30, 50 and 70 %.
- Averaging (Gbma) is kept if it is best at a majority of these designs, or if no variant beats it at any of them by more than 1 %.
- At block level (secondary): the averaged against the posterior-mode GP, as below.

**5. Block model and cutoff.** At block level (all blocks, CRPS as the mean CRPS per task, then geometric means over properties and fields;
coverage of 80 % intervals), with kappa = 1:
- The base model is kept unless an alternative has a lower CRPS for both field and zone means and a coverage at least as close to 0.80
  for both. The alternatives are: without the offset term; without the split of TabPFN's variance (all of v_T added as independent);
  without averaging; without the mean correction; and the correction at every n.
- Cutoff, at point level: the SSE ratio of the point GP's corrected mean over TabPFN's, pooled in the n bins
  [0, 25), [25, 50), [50, 100) and [100, ∞) on the 10, 30, 50, 70 and 90 % designs.
  - The correction starts at the smallest edge e in {25, 50, 100} from which every bin has a ratio below 0.98; it is never applied if
    there is no such edge.
  - The cutoff of 50 is kept if e = 50.

Reported also, without a rule: the same comparisons with kappa = 1.10; and every contrast by field, including LYD.112 and RUM.70 alone.

## Outputs

- `experiments/development/development.py` writes the block and point analyses and the report, `report_development.txt`, which goes to `expected/`.
- The TabPFN outputs behind choices 1–3 are read from their run folders.
