# Two further checks of the block model on the development fields (written 2026-11-07, before running them)

Both use the development splits of the 14 development fields, from the stored point outputs (no TabPFN refits), as choice 5 of
`plan_development.md`: CPU, this repository's code. Scoring as in `experiments/development/development.py`: coverage of 80 % intervals (field | zone) and CRPS ratio
variant / base (field | zone) with 95 % bootstrap intervals over fields, at kappa = 1 and kappa = 1.10.

**Check first.** `base`, the method with the mean correction from 25 samples, gives 6954 blocks and coverage 0.780 | 0.765 at kappa = 1 and
0.827 | 0.807 at kappa = 1.10. If it does not, stop.

**Reading.** A variant is "better on the rule" only if its CRPS is lower for both field and zone means and its coverage is at least as close
to 0.80 for both, at kappa = 1. Whatever the result, the method stays as it is and the 43 fields are not used.

## 1. The independent part from the GP's nugget (`experiments/development/dev_nugget.py`)

- `base`: V = V_fB + sum_{a in B} max(0, v_T(s_a) - v_bar) / m_B^2.
- `gp_nugget`: V = V_fB + sd(r)^2 tau2_bar / m_B, with tau2_bar the nugget mixed over the averaging grid, the same at every location. This is
  ordinary block kriging of the standardised residuals with the nugget kept at the target locations.
- `no_split`: all of v_T as independent.
- Everything else is identical. Secondary: the same contrasts by n (5-10, 15-20, 30 or more) and, for zones, by m_B (3-9, 10-29, 30 or more).
- Output: `expected/development/report_dev_nugget.txt`.

## 2. The squared mean error below 25 samples (`experiments/development/dev_mu2.py`)

Below 25 samples the mean of P_B is TabPFN's own block mean while its variance keeps the GP's. Under the model the mean squared difference
would be v_hat_B + mu_fB^2, and P_B omits mu_fB^2.

- `base`: the method.
- `add_mu2`: for n < 25, V = v_hat_B + mu_fB^2, with mu_fB the GP's block mean in the original units. Mean unchanged. Identical to `base` from
  25 samples.
- Reported on all blocks and on the blocks with n < 25, the only ones that differ. Secondary: n 5-10, 15-20, 25 or more.
- Output: `expected/development/report_dev_mu2.txt`.
