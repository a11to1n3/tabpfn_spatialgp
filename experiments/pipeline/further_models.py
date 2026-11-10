"""Further models on the splits of a set: the offset-only model (the shared-error-only model of the study's test 4) and the
difference estimators (exploratory).

Reads the saved point outputs of a set (block_cases.load_case); CPU only, no TabPFN refits, no R. One row per block and variant, in the layout
of the stored rows (unscaled predictive mean M, variance V, degrees of freedom df):
  TAB   TabPFN, errors independent (check against the stored rows)
  J1p   the chosen block model at kappa = 1, refitted (check against the stored rows)
  O1    offset only (sigma^2 = 0): M = mean m_T + [n >= GATE_N] rbar, V = s_r^2 / n + (1/m_B^2) sum v_T
  DE    difference estimator, field's residuals: M = mean m_T + rbar, V = s_r^2 (1/n + 1/m_B), df = n - 1
  DEz   difference estimator, the block's own revealed residuals (n_B >= 2): M = mean m_T + rbar_B, V = s_rB^2 (1/n_B + 1/m_B), df = n_B - 1
r = y - m_oof (cross-fitted residuals at the revealed locations); variances with divisor n - 1. Also one diagnostic row per case.

  python -m experiments.pipeline.further_models rows --set first|final --out DIR [--fields ...] [--workers 8] [--designs ...] [--reps ...]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core.block_model import PRIORS                               # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent   # noqa: E402

POP = PRIORS["POP"]
FILTER = {}                                                                   # --designs / --reps: for tests only


def case_rows(case):
    S, tr, te, y = case["S"], case["tr"], case["te"], case["y"]; n = len(tr); yq = y[te]
    blocks, revealed = block_cases.blocks_of(case)
    mT, sT = case["mT"], case["sT"]; gate = n >= F.GATE_N
    rows = []
    key = dict(set=case["set"], src="external" if case["field"] in F.C.EXTERNAL else "limesoda", field=case["field"],
               design=case["design"], rep=case["rep"], target=case["target"], n=n, m=len(te))

    def add(variant, b, M, V, df=np.inf, err=""):
        i = blocks[b]; failed = bool(err) or not (np.isfinite(M) and np.isfinite(V) and V > 1e-18)
        rows.append(dict(key, block=b, mB=len(i), nB=len(revealed[b]), variant=variant, M=float(M), V=float(V), df=float(df),
                         obs=float(yq[i].mean()), failed=failed, own_fallback=False,
                         err=err or ("non-finite mean or variance <= 1e-18" if failed else "")))

    for b, i in blocks.items():
        add("TAB", b, float(mT[i].mean()), float(np.sum(sT[i] ** 2)) / len(i) ** 2)
    diag = dict(key)
    if case["m_oof"] is None:
        for v in ("J1p", "O1", "DE", "DEz"):
            for b in blocks:
                add(v, b, np.nan, np.nan, err="no cross-fitted residuals")
        return rows, diag
    r = y[tr] - case["m_oof"]; rbar = float(r.mean()); s2 = float(r.var(ddof=1)); pos = {g: k for k, g in enumerate(tr)}
    diag.update(rbar=rbar, s_r=float(np.sqrt(s2)), sd_r=float(r.std()))
    # J1p: the chosen block model (block_model.block_model, written out to keep the fit for the diagnostics)
    try:
        L = field_extent(S); g = ResidualField(POP, L).fit(S[tr], r, block_range_floor(S[tr]))
        moments, vbar = g.block_moments(S[te], blocks); independent = np.clip(sT ** 2 - vbar, 0, None)
        for b, i in blocks.items():
            mb, vb = moments[b]
            add("J1p", b, (float(mT[i].mean()) + mb) if gate else float(mT[i].mean()), float(np.sum(independent[i])) / len(i) ** 2 + vb)
        share = lambda th: float(np.exp(th[0]) / (np.exp(th[0]) + np.exp(th[2])))   # noqa: E731
        diag.update(share_map=share(g.map_theta), range_rel_map=float(np.exp(g.map_theta[1]) / L),
                    share_w=float(np.dot(g.w, [share(t) for t in g.thetas])),
                    range_rel_w=float(np.exp(np.dot(g.w, [t[1] for t in g.thetas])) / L))
    except Exception as e:                                                    # counted and reported, never replaced
        for b in blocks:
            add("J1p", b, np.nan, np.nan, err=repr(e)[:150])
    for b, i in blocks.items():
        m = len(i); mTB = float(mT[i].mean()); vind = float(np.sum(sT[i] ** 2)) / m ** 2
        add("O1", b, mTB + (rbar if gate else 0.0), s2 / n + vind)
        add("DE", b, mTB + rbar, s2 * (1 / n + 1 / m), df=n - 1)
        rb = r[[pos[j] for j in revealed[b]]]
        if len(rb) >= 2:
            add("DEz", b, mTB + float(rb.mean()), float(rb.var(ddof=1)) * (1 / len(rb) + 1 / m), df=len(rb) - 1)
    return rows, diag


def field_rows(job):
    set_name, field, out_dir, flt = job
    FILTER.update(flt); t0 = time.monotonic(); rows, diags, missing = [], [], 0
    for f, data, design, rep, tr in block_cases.iter_splits(set_name, [field]):
        if (FILTER.get("designs") and design not in FILTER["designs"]) or (FILTER.get("reps") and rep not in FILTER["reps"]):
            continue
        for target in F.targets(f, set_name):
            if not np.all(np.isfinite(data["Y"][:, F.C.TARGETS.index(target)])):
                continue
            case = block_cases.load_case(set_name, f, data, design, rep, tr, target)
            if case is None:
                missing += 1; continue
            rr, d = case_rows(case); rows += rr; diags.append(d)
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"models_{set_name}_{field}.csv", index=False)
    pd.DataFrame(diags).to_csv(Path(out_dir) / f"diagnostics_{set_name}_{field}.csv", index=False)
    return field, len(rows), missing, time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["rows"])
    ap.add_argument("--set", required=True, choices=["first", "final"]); ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="*"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--designs", nargs="*", help="tests only"); ap.add_argument("--reps", nargs="*", type=int, help="tests only")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    flt = {"designs": a.designs, "reps": a.reps}
    fields = a.fields or sorted({f for group, _, _ in F.SETS[a.set] for f in group})
    jobs = [(a.set, f, str(out), flt) for f in fields]
    with Pool(min(a.workers, len(jobs))) as pool:
        for field, n, missing, sec in pool.imap_unordered(field_rows, jobs):
            print(f"{a.set} {field}: {n} rows, {missing} cases without point outputs, {sec:.0f} s", flush=True)


if __name__ == "__main__":
    main()
