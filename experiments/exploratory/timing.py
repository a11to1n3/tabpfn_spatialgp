"""Exploratory analysis E5 of docs/exploratory/plan_E3-E5.md: the cost of every method on the splits of expected/sensitivity/timing.csv (run alone on the GPU machine).

  python -m experiments.exploratory.timing <out>

Splits: the 20 fields of the factor-setting data without Oxford and Meuse, repeat 120, every sample size, SOC. TabPFN on the GPU, all
other computations on one CPU thread (R single-threaded), as in the study. Each method is timed from the revealed data to the field and
zone means; model loading and the first CUDA call are not timed. Writes <out>/timing_methods.csv and prints medians.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json                                                                   # noqa: E402
import subprocess                                                             # noqa: E402
import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

OUT = Path(sys.argv[1])
from experiments.core import config as F                                      # noqa: E402
from experiments.paths import CONFIGS                                         # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.sensitivity import folds                                     # noqa: E402  (make, cross_fitted: the study's TabPFN settings)
from experiments.core.common import bars, covariates, kriging, tabpfn        # noqa: E402
from experiments.core.block_model import PRIORS                               # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent   # noqa: E402
import torch                                                                  # noqa: E402

C = F.C
SHAPES = json.loads((CONFIGS / "fixed" / "shapes.json").read_text())
sync = torch.cuda.synchronize


def tabpfn_at(xa, yt, xq):
    """One fit of our TabPFN and its predictive moments and quantiles at xq (as the study's point outputs)."""
    bq = bars(folds.make().fit(xa, yt), xq); m, v = bq.moments(); bq.quantiles(np.array(C.QUANTILES)); sync()
    return m, v


def area_sums(m, v, blocks):
    return {b: (m[i].mean(), np.sum(v[i]) / len(i) ** 2) for b, i in blocks.items()}


def r_fix(S, tr, te, cols, column):
    st, out = block_cases.run_r("all", S, tr, te, cols, {"shapes": {column: {"range": SHAPES[column]["range_rel"] * field_extent(S),
                                                                       "nugget_share": SHAPES[column]["nugget_share"]}}, "parts": ["fix"]})
    return st, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(["Rscript", "-e", '.libPaths(c(Sys.getenv("SOILGP_RLIB"), .libPaths())); suppressMessages({library(gstat); library(automap)})'],
                   check=True, capture_output=True)
    r_start = time.perf_counter() - t0
    rows, warm = [], False
    for field, data, design, rep, tr in block_cases.iter_splits("final", list(C.LIMESODA)):
        if rep != 120:
            continue
        S = np.asarray(data["coordinates"], float); te = np.setdiff1d(np.arange(len(S)), tr)
        y = data["Y"][:, C.TARGETS.index("SOC_target")]; yt = y[tr]
        xa, xb = covariates(data, tr, te, with_coords=True)
        xa0, xb0 = covariates(data, tr, te, with_coords=False)
        blocks, revealed = block_cases.blocks_of(dict(S=S, tr=tr, te=te))          # the fixed zones (not timed: shared by all methods)
        if not warm:
            tabpfn_at(xa, yt, xb); warm = True
        rec = dict(field=field, design=design, n=len(tr), m=len(te), r_start_s=r_start)
        # TabPFN alone
        t = time.perf_counter(); m, v = tabpfn_at(xa, yt, xb); area_sums(m, v, blocks); rec["TabPFN"] = time.perf_counter() - t
        # TabPFN-GP: TabPFN, five fold fits, GP and area moments
        t = time.perf_counter(); m, v = tabpfn_at(xa, yt, xb)
        m_oof, v_oof = folds.cross_fitted(xa, yt, folds.S_.OOF_FOLDS); sync()
        g = ResidualField(PRIORS["POP"], field_extent(S)).fit(S[tr], yt - m_oof, block_range_floor(S[tr])); g.block_moments(S[te], blocks)
        rec["TabPFN-GP"] = time.perf_counter() - t
        # WH: TabPFN, five fold fits, correlogram (R) and area quadratic forms
        t = time.perf_counter(); m, v = tabpfn_at(xa, yt, xb)
        m_oof, v_oof = folds.cross_fitted(xa, yt, folds.S_.OOF_FOLDS); sync()
        z = (yt - m_oof) / np.sqrt(np.clip(v_oof, 1e-18, None))
        st, out = r_fix(S, tr, te, {"value": yt, "pred_oof": m_oof, "z_oof": z}, "z_oof")
        R = out.get("corr_z_fix"); sd = np.sqrt(v)
        if R is not None:
            for b, i in blocks.items():
                float(sd[i] @ R[np.ix_(i, i)] @ sd[i])
        rec["WH"] = time.perf_counter() - t; rec["WH_ok"] = st.get("z_fix") == "ok"
        # Block RK: TabPFN at all locations (in-sample means), block kriging of in-sample residuals (R)
        t = time.perf_counter(); mall, vall = tabpfn_at(xa, yt, np.vstack([xa, xb]))
        st, out = r_fix(S, tr, te, {"value": yt, "pred_ins": mall[:len(tr)]}, "r_ins")
        km, kc = out.get("rins_fix_mean"), out.get("rins_fix_cov")
        if km is not None:
            for b, i in blocks.items():
                float(km[i].mean()), float(kc[np.ix_(i, i)].mean())
        rec["Block RK"] = time.perf_counter() - t; rec["RK_ok"] = st.get("rins_fix") == "ok"
        # TabPFN-KpR: leave-one-out kriging features (R automap), one TabPFN fit without coordinates
        t = time.perf_counter(); ok = True
        try:
            k = kriging("kpr", S, tr, te, yt); fa, fb = k["kpr_training"], k["kpr_query"]
            xk, xq = np.column_stack([xa0, fa[["mean", "variance"]]]), np.column_stack([xb0, fb[["mean", "variance"]]])
            bq = bars(tabpfn().fit(xk, yt), xq); bq.moments(); bq.quantiles(np.array(C.QUANTILES)); sync()
        except Exception:
            ok = False
        rec["TabPFN-KpR"] = time.perf_counter() - t; rec["KpR_ok"] = ok
        # Sample mean
        t = time.perf_counter()
        for b, i in blocks.items():
            r = revealed[b]
            if len(r) >= 2:
                y[r].mean(), np.var(y[r], ddof=1)
        rec["Sample mean"] = time.perf_counter() - t
        rows.append(rec); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "timing_methods.csv", index=False)
    meth = ["TabPFN", "TabPFN-GP", "WH", "Block RK", "TabPFN-KpR", "Sample mean"]
    print(f"R start-up (Rscript loading gstat and automap): {r_start:.2f} s")
    print("median seconds per split by n (ratio to TabPFN):")
    for n, g in d.groupby("n"):
        print(f"  n={n:3d} ({len(g)} splits): " + "; ".join(f"{mm} {g[mm].median():.3f} ({(g[mm] / g['TabPFN']).median():.1f}x)" for mm in meth))
    print("all splits: " + "; ".join(f"{mm} {d[mm].median():.3f} [{d[mm].quantile(0.1):.3f}, {d[mm].quantile(0.9):.3f}]" for mm in meth))


if __name__ == "__main__":
    main()
