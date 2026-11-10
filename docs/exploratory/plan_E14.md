# E14: means over more locations (written 2026-10-27, before running it)

**Why.** Across fields, TabPFN-GP's coverage fell with the number of locations averaged:
- zones with 30 or more unrevealed locations: 0.770;
- Spearman correlation of per-field coverage with N: −0.39 for field means, −0.46 for zone means;
- lowest on Ontario, the field with the most locations.

Real field and zone means are means over dense prediction grids. Across fields, the number of locations is confounded with the field itself. E14 asks whether coverage falls with the number of locations averaged *within the same field and split*, and what that implies for means over denser grids. It is exploratory and is reported whatever it shows.

**Inputs.**
- The study's saved point outputs (`work/run/points_study`), CPU only, with the study's code unchanged.
- TabPFN-GP with the population prior, averaging and κ = 1.15, rebuilt as in `experiments/exploratory/further_models_rows.py`. Check: with all locations, M and V equal the saved J1p rows.

## Design

Each split of the 43 fields is used with every property. Each block is an area of the study:
- the field's unrevealed locations;
- each zone's unrevealed locations.

Within each block, the unrevealed locations are put in a random order, seeded by crc32(field|design|rep|target|block). The nested sub-blocks are the first m locations of that order, for m in {10, 30, 100, 200}, whenever m is smaller than the block's size m_B. The block itself is the level "all".

A sub-block is a coarser sample of the same area, so moving from m = 10 to "all" is the step from a coarse to a dense set of locations within one area.

**Methods on every sub-block, with the study's formulas and factors:**
- TabPFN-GP (κ = 1.15);
- the shared-error-only model (κ = 1.15);
- TabPFN, combined as independent (factor 2.40);
- the sample mean: t interval with n_B − 1 degrees of freedom, variance s²(1/n_B + 1/m), factor 1.05, where n_B ≥ 2.

Block RK, TabPFN-KpR and WH are left out, because they need R or TabPFN refits.

## Reported (80 % intervals; bootstrap over fields, 10,000 resamples, seed 0)

1. **Coverage by level, field means.** For each method, coverage at each level m over the splits whose field block has more than m locations, and at "all" over the same splits.
   - The paired difference D_m = coverage("all") − coverage(m = 10), on the splits with m_B > 10, with a 95 % interval.
   - The same by class of m_B: ≤ 100, 101–200 and > 200.
2. **Coverage by level, zone means.** As in 1, for zones with m_B > 10 (levels 10, 30, "all") and with m_B > 30 (levels 30, "all").
3. **Dense limit.** For TabPFN-GP at "all":
   - the share of its block variance from the independent part, Σ_i ind_i / m² over V, by class of m_B, for field and zone means;
   - coverage with that part removed, V_∞ = V − Σ_i ind_i / m², by class of m_B.

   On a denser grid over the same area, the independent part shrinks towards 0, so V_∞ is the variance TabPFN-GP would give for a mean over infinitely many locations. Scoring it against the observed mean over the m_B locations gives a lower bound on the coverage over a dense grid: that observed mean still contains an average of independent errors, with variance about Σ ind / m², which a dense mean would not contain.

**Interpretation, fixed before running.**
- If TabPFN-GP's D_m for field means and zone means has a 95 % interval that includes 0, or lies above −0.03, coverage does not fall with the number of locations within an area. The across-field pattern then reflects which fields are large.
- If D_m lies below −0.03, coverage falls within areas, and means over dense grids are expected to be covered less often. The dense-limit coverage then estimates how much less.
