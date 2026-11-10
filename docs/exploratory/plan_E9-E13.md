# Exploratory analyses E9–E13 (written before running them)

**Status.** They were specified after the study's results had been seen, so all of them are exploratory. Each is run once and reported whatever it shows.

**Inputs.** Unless stated otherwise:
- the saved study outputs: `work/rows`, `work/run/points_study`;
- the factor-setting data: `repro work/factor_setting/rows_first_*.csv`, `repro work/run/points_first`;
- the study's code, unchanged.

**One factor rule for everything.** Every factor below is set by the rule of the paper: the smallest factor on 0.50–5.00 (step 0.05) whose mean of pooled field- and zone-mean 80 % coverage on the factor-setting data reaches 0.80. Each external task is counted once (Oxford without colour; Meuse SOC only), as for κ = 1.15. The CRPS version (with or without the factor) is the one with the lower CRPS on the factor-setting LimeSoDa rows.

## E9. Inference that respects datasets and groups (saved rows)

1. **Two-stage inference over the 17 source datasets.** The datasets are:
   - SOIL250; SOJA98;
   - SensRes; Midwest;
   - Ontario; Czech STEROPES; Italian STEROPES; Cook farm; Agroscope;
   - Auzeville; Epoisses; Ploudaniel; Crouel; Maricopa (three fields); Walnut Creek; EIDC (Henfaes, North Wyke); Caroline.

   Each of the five tests is recomputed with bootstrap intervals over datasets (10,000 resamples, seed 0) instead of over fields. Test 5 uses one-sided exact Wilcoxon tests over the 17 dataset means of the field log-ratios, Holm-corrected over the ten contrasts.
2. **Leave-one-group-out.** Each test is recomputed without each of the five groups in turn. Reported: the range of each estimate and the number of the five omissions in which each criterion is met.
3. **Per-group test quantities:**
   - test 2: D with 90 % intervals;
   - test 3: coverage;
   - test 4a: shared-error-only field coverage;
   - test 4b: the zone-coverage difference;
   - test 5: CRPS ratios.
4. **Field-balanced rise.** For the fields with n = 5, 10, 30 and 50 all present, D = coverage at {30, 50} minus coverage at {5, 10}, computed per field and averaged with equal field weights, with a bootstrap over fields. Reported for every method.
5. **Coverage by size:**
   - TabPFN-GP's coverage by field-size tertile (N);
   - by convex-hull area class (< 1, 1–10, 10–100, > 100 ha; uniformity trials excluded);
   - for zones, by m_B class (3–9, 10–29, ≥ 30);
   - the Spearman correlation of per-field coverage with N.
6. **By covariate source.** Fields with released covariates against fields with public covariates only: TabPFN-GP's coverage, and its CRPS ratio to the sample mean and to TabPFN.
7. **TabPFN-GP at n = 5 on all blocks,** not only those where TabPFN-KpR returned a prediction.
8. **Share of the summed TabPFN-GP CRPS by sample size,** field and zone means.

## E10. Factors by one rule for all methods (saved rows; factor-setting rows)

Factors by the rule above for TabPFN, TabPFN-KpR, block RK, WH and the sample mean, and also for:
- the shared-error-only model (its rows on the factor-setting data are computed from the saved point outputs);
- the difference estimators (field: DE, mean correction at every n, t interval with n − 1 degrees of freedom; zone: DEz, within-zone residual mean).

The five tests and the CRPS ratios are recomputed with these factors (TabPFN-GP: κ = 1.15). The zone CRPS ratio of the shared-error-only model to TabPFN-GP is also reported.

## E11. Further comparators and component tests (saved points; new rows on the factor-setting data and on the 43 fields)

| Code | Model |
|---|---|
| RKo | Block RK of cross-fitted residuals (variant RK_oof_fix_v, already in the saved rows) |
| WHl | WH with a correlogram fitted to the field's own errors (WH_loc, saved rows) |
| KpRp | TabPFN-KpR with its published TabPFN settings (KpR, saved rows) |
| Z | Zone-effect model: the error model f(s) = β + u_z(s) with independent zone effects u_z ~ N(0, σ_u²) instead of the GP's spatial part, fitted by REML to the same standardised held-out errors, then combined with TabPFN exactly as TabPFN-GP is (mean correction from n = 25; TabPFN's variance minus the model's leave-one-out latent variance) |
| GP-np | TabPFN-GP without the cross-field prior (flat prior), with averaging |
| GP-na | TabPFN-GP with the prior, without averaging (the posterior mode only) |
| GP-nn | TabPFN-GP with neither (REML mode, flat prior) |
| GP-x | TabPFN-GP with the prior refitted without SL.125 and UL.120, whose released coordinates are dummies |

Each model gets its own factor by the rule (the rows of the new models on the factor-setting data are computed from the saved point outputs). On the 43 fields, reported:
- coverage of field and zone means;
- D;
- CRPS ratio of TabPFN-GP to each (field and zone means), with 95 % bootstrap intervals over fields.

## E12. All methods with five seeds (GPU)

**What is rerun.** Seeds 1–4 for TabPFN (the seed of TabPFN and of the fold split, set exactly as seed 0 was), for every method that uses TabPFN: TabPFN-GP, the shared-error-only model, TabPFN, TabPFN-KpR, block RK and WH. The rerun covers:
- all 43 fields, all splits;
- the study's code, with the seed changed through its settings and the default of `common.tabpfn`.

**Reported:**
- the five tests for each seed;
- each comparator's CRPS ratio for each seed, and the geometric mean over the five seeds;
- the range of TabPFN-GP's coverage over the seeds.

**Check.** Seed 0 is the study's run.

**Added after the four reruns had finished, before computing it.** E8's measures (`plan_E7-E8.md`) for TabPFN-GP on all 43 fields and seeds 0–4: the standard deviation over the seeds of each interval endpoint divided by the half-width (median and 90th percentile, by n), coverage for each seed, and the CRPS of each seed relative to seed 0 (`experiments/exploratory/seeds_spread.py`).

## E13. Comparators with a factor for each sample size, set on the factor-setting data (saved rows; added before running it)

For TabPFN, TabPFN-KpR, block RK, WH and the sample mean, a factor for each n in {5, 10, 15, 20, 30, 50, 75, 100} is set by the rule above, applied to the factor-setting rows with that n (each external task once). Each comparator keeps its CRPS version.

On the 43 fields, with these factors and TabPFN-GP with κ = 1.15, reported:
- coverage by n and D for field and zone means;
- the CRPS ratio of TabPFN-GP to each comparator, on the common blocks.

This answers whether a sample-size-dependent factor that a practitioner could set from other fields removes the rise in the comparators' coverage. The E2 analysis, by contrast, tuned the factors on the evaluation results themselves.
