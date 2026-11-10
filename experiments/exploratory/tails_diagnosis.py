"""Diagnosis of the interval tails on the factor-setting data (first-set rows), before any fix is chosen.

  python -m experiments.exploratory.tails_diagnosis <rows dir with rows_first_*.csv>

TabPFN-GP (variant J1p, stored with kappa = 1), each external task counted once (as for kappa = 1.15). Coverage of central 50/80/90/95 %
intervals with kappa = 1.15 and normal quantiles, by support (field, zone) and by n; quantiles of |obs - M| / V^(1/2) and their ratio to
the 80 % quantile (normal: 0.526, 1, 1.283, 1.529).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, kurtosis

d = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(Path(sys.argv[1]).glob("rows_first_*.csv"))], ignore_index=True)
print("variants:", sorted(d.variant.unique()))
d = d[(d.variant == "J1p") & ~d.failed.astype(bool)]
d = d[~((d.field == "OXFNC") | ((d.field == "MEUSE") & (d.target != "SOC_target")))]
d["kind"] = np.where(d.block == "field", "field", "zone")
d["z"] = (d.obs - d.M) / np.sqrt(d.V)
print(f"rows: {len(d)}; fields: {d.field.nunique()}; n values: {sorted(d.n.unique())}")
LEV = (0.5, 0.8, 0.9, 0.95)
for kind in ("field", "zone"):
    s = d[d.kind == kind]
    print(f"\n== {kind} means ({len(s)} rows), coverage with kappa 1.15 at 50/80/90/95 %, and |z| quantile / 80 % quantile:")
    for label, sel in [("all", s.n > 0)] + [(f"n={n}", s.n == n) for n in sorted(s.n.unique())]:
        q = s[sel]; a = np.abs(q.z.to_numpy())
        cov = [np.mean(a <= norm.ppf((1 + l) / 2) * 1.15) for l in LEV]
        qs = np.quantile(a, LEV); r = qs / qs[1]
        print(f"  {label:6s} ({len(q):6d}) cov " + " ".join(f"{c:.3f}" for c in cov) + "   ratio " + " ".join(f"{x:.3f}" for x in r)
              + f"   excess kurtosis {kurtosis(q.z):.2f}")
    # zones with and without a revealed sample
    if kind == "zone":
        for lab, sel in (("zones with a sample", s.nB > 0), ("zones without", s.nB == 0)):
            a = np.abs(s[sel].z.to_numpy())
            print(f"  {lab:20s} ({sel.sum()}) cov " + " ".join(f"{np.mean(a <= norm.ppf((1 + l) / 2) * 1.15):.3f}" for l in LEV))
print("\nnormal ratios:", " ".join(f"{norm.ppf((1 + l) / 2) / norm.ppf(0.9):.3f}" for l in LEV))
