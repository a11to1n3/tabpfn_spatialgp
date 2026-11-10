"""E12 of docs/exploratory/plan_E9-E13.md, the measures of E8 on all 43 fields (CPU): spread of TabPFN-GP's intervals over seeds 0-4, kappa = 1.15.

  python -m experiments.exploratory.seeds_spread <work> <out>  -> <out>/seeds_spread.txt

Reads <work>/rows (seed 0, the study) and <work>/rows_seed1..4 (run_seeds.sh). TabPFN-GP's rows (variant J1p) on the areas scored with
all five seeds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
KAPPA, Z = 1.15, norm.ppf(0.9)
KEY = ["field", "design", "rep", "target", "block"]
DIRS = {0: WORK / "rows", **{s: WORK / f"rows_seed{s}" for s in (1, 2, 3, 4)}}
lines = []


def say(s=""):
    lines.append(s); print(s, flush=True)


def crps(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def load(seed):
    parts = []
    for p in sorted(DIRS[seed].glob("rows_study_*.csv")):
        r = pd.read_csv(p, low_memory=False, usecols=KEY + ["n", "variant", "M", "V", "obs", "failed"])
        parts.append(r[(r.variant == "J1p") & ~r.failed.astype(bool)].drop(columns=["variant", "failed"]))
    return pd.concat(parts, ignore_index=True).assign(seed=seed)


d = pd.concat([load(s) for s in DIRS], ignore_index=True)
d["kind"] = np.where(d.block == "field", "field", "zone")
sd = KAPPA * np.sqrt(d.V); d["lo"], d["hi"] = d.M - Z * sd, d.M + Z * sd
d["covered"] = ((d.obs >= d.lo) & (d.obs <= d.hi)).astype(float); d["crps"] = crps(d.obs, d.M, sd)
complete = d.groupby(KEY).seed.nunique().pipe(lambda s: s[s == 5]).index
g = d.set_index(KEY).loc[complete].reset_index()
say("=== E12 (docs/exploratory/plan_E9-E13.md), the measures of E8 on all 43 fields: TabPFN-GP over seeds 0-4, kappa = 1.15 ===")
say(f"fields {g.field.nunique()}; areas with all five seeds {len(complete)} of {d.groupby(KEY).ngroups}")
for kind in ("field", "zone"):
    q = g[g.kind == kind]
    agg = q.groupby(KEY).agg(lo_sd=("lo", "std"), hi_sd=("hi", "std"), hi_m=("hi", "mean"), lo_m=("lo", "mean"), n=("n", "first"))
    half = (agg.hi_m - agg.lo_m) / 2
    r = np.concatenate([agg.lo_sd / half, agg.hi_sd / half]); nn = np.concatenate([agg.n, agg.n])
    say(f"\n{kind} means ({len(agg)} areas):")
    say(f"  sd of an interval endpoint over the seeds / half-width: median {np.median(r):.3f}, 90th percentile {np.percentile(r, 90):.3f}")
    for n in sorted(set(nn)):
        rr = r[nn == n]
        say(f"    n = {n:3d}: median {np.median(rr):.3f}, 90th percentile {np.percentile(rr, 90):.3f}")
    cov = q.groupby("seed").covered.mean()
    say("  coverage by seed: " + "; ".join(f"{s}: {v:.3f}" for s, v in cov.items()) + f"  (range {cov.max() - cov.min():.3f})")
    c0 = q[q.seed == 0].crps.sum()
    say("  CRPS relative to seed 0: " + "; ".join(f"{s}: {q[q.seed == s].crps.sum() / c0:.3f}" for s in range(1, 5)))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "seeds_spread.txt").write_text("\n".join(lines) + "\n")
