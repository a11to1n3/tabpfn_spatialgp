# E15–E17: the factor, zones without a sample, zones with few samples (written 2026-11-08, before running them)

**Status.** Specified after the study's results had been seen, so exploratory. Each is run once and reported whatever it shows.

**Inputs.** The saved rows of the 43 fields (`work/rows`, `work/variants`) and the factors of E10 (`expected/exploratory/factors_once.json`), CPU
only, with the study's scoring code unchanged. No new predictions.

**Check first.** TabPFN-GP's coverage 0.829 | 0.794 and 0.756 in zones without a sample (the study's report); the zone-effect model's
coverage 0.834 | 0.781 and that without the cross-field prior 0.820 | 0.785 (E11). If not reproduced, stop.

## E15. TabPFN-GP without its factor
TabPFN-GP with κ = 1 on the areas scored by all methods: coverage (field | zone), D (field | zone), coverage in zones without a revealed
sample, and the CRPS ratio TabPFN-GP with κ = 1.15 (as in the study) / TabPFN-GP with κ = 1, with 95 % intervals over fields.

## E16. Zones without a revealed sample (n_B = 0), further models
On the areas where TabPFN-GP and the model are both scored (as in E11): coverage with each model's own factor and the CRPS ratio
TabPFN-GP / model (both by the version rule, as in E11), for the zone-effect model, the four TabPFN-GP variants without components,
block RK of held-out errors and WH with each field's own correlogram; TabPFN-GP on the same zones and the shared-error-only model
(κ = 1.15, and its own factor) for reference.

## E17. Zones by number of revealed samples
Zones with n_B = 1, n_B = 2 and n_B ≥ 3: number of zone means; coverage of TabPFN-GP, the four TabPFN-based comparators (the study's
factors), the shared-error-only model and the zone-effect model; for n_B ≥ 2 also the sample mean and the within-zone difference
estimator (coverage only: with one degree of freedom their CRPS is infinite); CRPS ratio TabPFN-GP / comparator on the areas scored by
all methods (each method's CRPS version, as in the study), and TabPFN-GP / zone-effect model as in E11.

## Reading (fixed before running)
- E15: reported as a row of the table of further methods; its CRPS column is TabPFN-GP with κ = 1.15 / TabPFN-GP with κ = 1.
- E16: if the zone-effect model or a variant covers zones without a sample at least as close to 0.80 as TabPFN-GP does and its CRPS
  there is no higher, the paper's guidance for these zones names it alongside TabPFN-GP.
- E17: the guidance for zone means is extended to zones with one or two samples only if TabPFN-GP's coverage for n_B = 1 and for
  n_B = 2 each lies within [0.75, 0.85] and its CRPS is lower than each TabPFN-based comparator's with the 95 % interval below 1.
