"""Candidate fixes of the interval tails, compared on the factor-setting data only (first-set rows; each external task once).

  python -m experiments.exploratory.tails_fixes <rows dir with rows_first_*.csv>

Half-width of the central interval at level a (p = (1 + a) / 2), V the block variance stored with kappa = 1:
  current  kappa * z_p * V^(1/2),  kappa = 1.15
  C1       kappa_a * z_p * V^(1/2), kappa_a = the factor rule at level a (smallest factor on 0.50-5.00, step 0.05, whose mean of pooled
           field- and zone-mean coverage reaches a; at a = 0.80 it must return 1.15)
  C2       kappa * z_0.9 * V^(1/2) * t_nu(p) / t_nu(0.9), nu = n - 1: Student-t shape with n - 1 degrees of freedom, 80 % interval unchanged
  C3       C2's shape with a factor per level set by the same rule
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

d = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(Path(sys.argv[1]).glob("rows_first_*.csv"))], ignore_index=True)
d = d[(d.variant == "J1p") & ~d.failed.astype(bool)]
d = d[~((d.field == "OXFNC") | ((d.field == "MEUSE") & (d.target != "SOC_target")))]
field = (d.block == "field").to_numpy(); n = d.n.to_numpy(); a = np.abs(d.obs - d.M).to_numpy() / np.sqrt(d.V.to_numpy())
LEV = (0.5, 0.8, 0.9, 0.95); GRID = np.round(np.arange(0.50, 5.0001, 0.05), 2); K = 1.15
shape_t = lambda p: tdist.ppf(p, n - 1) / tdist.ppf(0.9, n - 1) * norm.ppf(0.9)        # noqa: E731


def rule(q, lev):
    """Smallest grid factor f with mean of pooled field and zone coverage of a <= f * q reaching lev."""
    for f in GRID:
        c = a <= f * q
        if (c[field].mean() + c[~field].mean()) / 2 >= lev:
            return float(f)
    return float(GRID[-1])


half = {"current": {}, "C1": {}, "C2": {}, "C3": {}}; fac = {"C1": {}, "C3": {}}
for lev in LEV:
    p = (1 + lev) / 2
    half["current"][lev] = K * norm.ppf(p) * np.ones_like(a)
    fac["C1"][lev] = rule(norm.ppf(p) * np.ones_like(a), lev); half["C1"][lev] = fac["C1"][lev] * norm.ppf(p) * np.ones_like(a)
    half["C2"][lev] = K * shape_t(p)
    fac["C3"][lev] = rule(shape_t(p), lev); half["C3"][lev] = fac["C3"][lev] * shape_t(p)
print("factors by level:", {k: v for k, v in fac.items()})
bands = {"n<=10": n <= 10, "13-25": (n >= 13) & (n <= 25), "n>=30": n >= 30, "all": n > 0}
for kind, sel in (("field", field), ("zone", ~field)):
    print(f"\n== {kind} means: coverage at 50 / 80 / 90 / 95 %")
    for m in half:
        print("  " + m.ljust(8) + "  ".join(f"{b} " + " ".join(f"{np.mean(a[sel & bs] <= half[m][l][sel & bs]):.3f}" for l in LEV)
                                         for b, bs in bands.items()))
    # widths relative to current at 95 %, median over rows by band
    for m in ("C1", "C2", "C3"):
        print("  " + f"{m} width / current at 95 %: " + "  ".join(f"{b} {np.median(half[m][0.95][sel & bs] / half['current'][0.95][sel & bs]):.3f}"
                                                                   for b, bs in bands.items()))
