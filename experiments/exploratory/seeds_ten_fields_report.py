"""E8 of docs/exploratory/plan_E7-E8.md (CPU): spread of TabPFN-GP's intervals over seeds 0-4 on ten study fields, kappa = 1.15.

  python -m experiments.exploratory.seeds_ten_fields_report <seeds dir> <work>/rows <out>  -> <out>/seeds_report.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SEEDS, ROWS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
KAPPA, Z = 1.15, norm.ppf(0.9)
KEY = ["field", "design", "rep", "target", "block"]
lines = []


def say(s=""):
    lines.append(s); print(s, flush=True)


def crps(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


d = pd.concat([pd.read_csv(p) for p in sorted(SEEDS.glob("seeds_*.csv"))], ignore_index=True)
d = d[~d.failed.astype(bool)].copy()
d["kind"] = np.where(d.block == "field", "field", "zone")
sd = KAPPA * np.sqrt(d.V); d["lo"], d["hi"] = d.M - Z * sd, d.M + Z * sd
d["covered"] = ((d.obs >= d.lo) & (d.obs <= d.hi)).astype(float); d["crps"] = crps(d.obs, d.M, sd)
say("=== E8 (docs/exploratory/plan_E7-E8.md): TabPFN-GP over seeds 0-4 of TabPFN and of the fold split, kappa = 1.15 ===")
say(f"fields: {', '.join(sorted(d.field.unique()))}; designs {sorted(d.design.unique())}; rows {len(d)}")

# check: seed 0 reproduces the study's rows
st = pd.concat([pd.read_csv(ROWS / f"rows_study_{f}.csv", low_memory=False) for f in d.field.unique()], ignore_index=True)
st = st[st.variant == "J1p"][KEY + ["M", "V"]]
m = d[d.seed == 0].merge(st, on=KEY, suffixes=("", "_s"))
say(f"check, seed 0 against the study rows: {len(m)} of {int((d.seed == 0).sum())} matched; largest relative difference "
    f"M {float((abs(m.M - m.M_s) / abs(m.M_s)).max()):.1e}, V {float((abs(m.V - m.V_s) / m.V_s).max()):.1e}")

complete = d.groupby(KEY).seed.nunique().pipe(lambda s: s[s == 5]).index
g = d.set_index(KEY).loc[complete].reset_index()
for kind in ("field", "zone"):
    q = g[g.kind == kind]
    agg = q.groupby(KEY).agg(lo_sd=("lo", "std"), hi_sd=("hi", "std"), half=("hi", "mean"), lo_m=("lo", "mean"))
    half = (agg.half - agg.lo_m) / 2
    r = np.concatenate([agg.lo_sd / half, agg.hi_sd / half])
    say(f"\n{kind} means ({len(agg)} areas with all five seeds):")
    say(f"  sd of an interval endpoint over the seeds / half-width: median {np.median(r):.3f}, 90th percentile {np.percentile(r, 90):.3f}")
    for design in sorted(q.design.unique(), key=lambda s: int(s[1:])):
        a = agg[agg.index.get_level_values("design") == design]; h = (a.half - a.lo_m) / 2
        rr = np.concatenate([a.lo_sd / h, a.hi_sd / h])
        say(f"    {design}: median {np.median(rr):.3f}, 90th percentile {np.percentile(rr, 90):.3f} ({len(a)} areas)")
    cov = q.groupby("seed").covered.mean()
    say("  coverage by seed: " + "; ".join(f"{s}: {v:.3f}" for s, v in cov.items()) + f"  (range {cov.max() - cov.min():.3f})")
    c0 = q[q.seed == 0].crps.sum()
    say("  CRPS relative to seed 0: " + "; ".join(f"{s}: {q[q.seed == s].crps.sum() / c0:.3f}" for s in range(1, 5)))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "seeds_report.txt").write_text("\n".join(lines) + "\n")
