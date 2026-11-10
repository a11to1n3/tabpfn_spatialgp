"""Exploratory: TabPFN's cross-fitted residuals with other numbers of folds, and the time of TabPFN-GP's computations (GPU);
sensitivity.py uses the residuals.

  python -m experiments.sensitivity.folds points --folds 10|loo|5 --out DIR [--fields ...] [--designs ...] [--reps ...]
      Cross-fitted TabPFN means and variances at the training rows of the final splits of the 20 fields, computed as point_model.our_model
      computes them with J = min(5, n) folds (same TabPFN settings, inputs, seed and fold assignment), but with J = min(10, n) folds or
      one fold per training row (leave-one-out): DIR/<folds>/<design>/<field>/<target>/rep<r>.npz with train_rows, m_oof and v_oof.
      Existing files are skipped, so an interrupted run is resumed by running it again. --folds 5 recomputes the stored residuals,
      as a check (sensitivity.py report compares them).
  python -m experiments.sensitivity.folds timing --out DIR [--fields ...] [--device cuda|cpu]
      The time of TabPFN-GP's computations for one split and property (replicate 120, SOC or organic matter) on one process: the final
      TabPFN fit with its predictive distribution at the unrevealed locations, the J = min(5, n) fold fits with their predictions, and
      the GP and block computations (CPU, one thread) -> DIR/timing.csv (DIR/timing_cpu.csv with --device cpu: TabPFN on one CPU thread).
      Run it alone on the machine.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one thread, as in the study's runs
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import time                                                                   # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from sklearn.model_selection import KFold                                     # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core.common import bars, covariates, point_file, save_npz, tabpfn  # noqa: E402
from experiments.core.block_model import PRIORS                               # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent   # noqa: E402

C = F.C
FOLDS = {"5": S_.OOF_FOLDS, "10": 10, "loo": None}


DEVICE = "cuda"


def make():
    if DEVICE == "cpu":                                                       # the same settings on one CPU thread (timing only)
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device="cpu", n_estimators=4, random_state=S_.RANDOM_STATE, inference_config=S_.INFERENCE_CONFIG)
    return tabpfn(n_estimators=4, target_transform=False)                     # our TabPFN settings (point_model.our_model)


def cross_fitted(xa, yt, J):
    """point_model.our_model's fold loop with J folds (None: one per training row)."""
    n = len(yt); m, v = np.empty(n), np.empty(n)
    for a, b in KFold(n if J is None else min(J, n), shuffle=True, random_state=S_.RANDOM_STATE).split(xa):
        if np.ptp(yt[a]) == 0:                                                # constant training fold: predict the constant
            m[b], v[b] = yt[a][0], max(float(np.var(yt)), 1e-12); continue
        m[b], v[b] = bars(make().fit(xa[a], yt[a]), xa[b]).moments()
    return m, v


def points(folds, out, fields, designs, reps):
    root = Path(out) / folds
    for field, data, design, rep, tr in block_cases.iter_splits("final", fields or list(C.LIMESODA)):
        if field not in C.LIMESODA or (designs and design not in designs) or (reps and rep not in reps):
            continue                                                          # the 20 fields only
        te = np.setdiff1d(np.arange(len(data["coordinates"])), tr); xa, _ = covariates(data, tr, te, with_coords=True)
        for target in F.targets(field, "final"):
            path = point_file(root, "ours", design, field, target, rep)
            y = data["Y"][:, C.TARGETS.index(target)]
            if path.exists() or not np.all(np.isfinite(y)):
                continue
            t0 = time.monotonic(); m, v = cross_fitted(xa, y[tr], FOLDS[folds])
            save_npz(path, train_rows=tr, m_oof=m, v_oof=v)
            print(f"{folds} {design} {field} {target} rep={rep} n={len(tr)} {time.monotonic() - t0:.1f} s", flush=True)


def timing(out, fields):
    import torch
    sync = torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
    if DEVICE == "cpu":
        from experiments.core.common import single_thread
        single_thread()
    rows, warm = [], False
    for field, data, design, rep, tr in block_cases.iter_splits("final", fields or list(C.LIMESODA)):
        if rep != 120:
            continue
        te = np.setdiff1d(np.arange(len(data["coordinates"])), tr); S = np.asarray(data["coordinates"], float)
        xa, xb = covariates(data, tr, te, with_coords=True); y = data["Y"][:, C.TARGETS.index("SOC_target")]; yt = y[tr]
        if not warm:                                                          # model loading and CUDA start-up, not timed
            bars(make().fit(xa, yt), xb).moments(); sync(); warm = True
        t0 = time.perf_counter()
        bq = bars(make().fit(xa, yt), xb); bq.moments(); bq.quantiles(np.array(C.QUANTILES)); sync()
        t1 = time.perf_counter()
        m_oof, _ = cross_fitted(xa, yt, S_.OOF_FOLDS); sync()
        t2 = time.perf_counter()
        case = dict(S=S, tr=tr, te=te)
        blocks, _ = block_cases.blocks_of(case)
        g = ResidualField(PRIORS["POP"], field_extent(S)).fit(S[tr], yt - m_oof, block_range_floor(S[tr])); g.block_moments(S[te], blocks)
        t3 = time.perf_counter()
        rows.append(dict(field=field, design=design, n=len(tr), m=len(te), final_fit_s=t1 - t0, fold_fits_s=t2 - t1, gp_blocks_s=t3 - t2))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows); d.to_csv(Path(out) / ("timing.csv" if DEVICE == "cuda" else "timing_cpu.csv"), index=False)
    print(summary(d))


def summary(d):
    """Medians over the splits (seconds), overall and for n <= 20 and n >= 30."""
    d = d.assign(total_s=d.final_fit_s + d.fold_fits_s + d.gp_blocks_s); cols = ["final_fit_s", "fold_fits_s", "gp_blocks_s", "total_s"]
    out = [f"{len(d)} splits, medians in seconds: " + ", ".join(f"{c} {d[c].median():.2f}" for c in cols)]
    for lab, s in (("n <= 20", d[d.n <= 20]), ("n >= 30", d[d.n >= 30])):
        out.append(f"  {lab} ({len(s)}): " + ", ".join(f"{c} {s[c].median():.2f}" for c in cols))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["points", "timing"]); ap.add_argument("--out", required=True)
    ap.add_argument("--folds", choices=list(FOLDS)); ap.add_argument("--fields", nargs="*")
    ap.add_argument("--designs", nargs="*", help="small tests only"); ap.add_argument("--reps", nargs="*", type=int, help="small tests only")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    global DEVICE
    DEVICE = a.device
    if a.stage == "points":
        points(a.folds, a.out, a.fields, a.designs, a.reps)
    else:
        timing(a.out, a.fields)


if __name__ == "__main__":
    main()
