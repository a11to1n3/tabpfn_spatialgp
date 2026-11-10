"""Core of the factor setting and of the study: loading the saved point outputs of each set, the R bridge, and the block predictions of every method.

A 'case' is one split x property: training rows tr, test rows te (the unrevealed locations), TabPFN's outputs and the KpR outputs.
case_rows(case, shapes) returns one row per block x method with the predictive distribution's mean M, variance V and degrees of freedom
df (inf = normal; finite = location-scale t, the sample mean), the observed block mean obs, and whether the method failed.
Scores and calibration factors are applied later (scoring.py), so every row keeps the unscaled V.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.core import config as F
from experiments.core.config import C
from experiments.core.common import interval_sd, load_field, point_file, training_rows
from experiments.core.block_model import PRIORS, block_model             # the block model J1p
from tabpfn_spatialgp import settings as S_
from tabpfn_spatialgp.blocks import zone_blocks
from tabpfn_spatialgp.gp_field import field_extent

BLOCK_KRIGING_R = Path(__file__).resolve().parent / "block_kriging.R"
FINAL_POINTS = F.POINTS["final"]                             # point outputs of the final set


# ------------------------------------------------------------------------------------------------------------------------- splits
def iter_splits(set_name, fields=None, min_test=None):
    """(field, data, design, rep, tr) of a set, in a fixed order; designs that leave fewer than 10 unrevealed locations are skipped."""
    for group_fields, designs, reps in F.SETS[set_name]:
        for field in group_fields:
            if fields and field not in fields:
                continue
            data = load_field(field)
            for design in designs:
                for rep in reps:
                    try:
                        tr = training_rows(data, field, design, rep, F.MIN_TEST if min_test is None else min_test)
                    except ValueError:
                        continue
                    yield field, data, design, rep, tr


# ------------------------------------------------------------------------------------------------------------------------- loading
def _load(path):
    return dict(np.load(path, allow_pickle=True)) if Path(path).exists() else None


def _paths(set_name, field, design, target, rep):
    """Files of one case (written by pipeline/points.py): ours (our TabPFN, with the cross-fitted and in-sample means), KpR (published settings)
    and KpR4 (our TabPFN settings)."""
    root = F.POINTS[set_name] if set_name in ("dev", "first") else FINAL_POINTS
    return {m: point_file(root, m, design, field, target, rep) for m in ("ours", "KpR", "KpR4")}


def load_case(set_name, field, data, design, rep, tr, target):
    """Everything one case needs, or None if the point outputs of our TabPFN are missing. Training rows are checked against the files."""
    p = _paths(set_name, field, design, target, rep)
    ours = _load(p["ours"])
    if ours is None:
        return None
    assert np.array_equal(ours["train_rows"], tr), (field, design, rep, target)
    te = np.setdiff1d(np.arange(len(data["coordinates"])), tr)
    y = data["Y"][:, C.TARGETS.index(target)]
    m_ins = ours["m_ins"]
    case = dict(set=set_name, field=field, design=design, rep=rep, target=target, tr=tr, te=te, y=y,
                S=np.asarray(data["coordinates"], float), mT=ours["none_mean"], sT=interval_sd(ours["none_quantiles"]),
                m_oof=ours.get("bb_m_oof"), v_oof=ours.get("bb_v_oof"), m_ins=m_ins, ours=ours)
    for m in ("KpR", "KpR4"):
        k = _load(p[m])
        if k is not None:
            assert np.array_equal(k["test_rows"], te)
            meta = json.loads(str(k["metadata"]))
            case[m] = (k["mean"], interval_sd(k["quantiles"]), meta.get("kriging_failed") not in (False, None))
    return case


# ------------------------------------------------------------------------------------------------------------------------- R bridge
def run_r(mode, S, tr, te, columns, params=None):
    """Run block_kriging.R in a temporary folder. columns: {name: values at the training rows}. Returns (status, {output name: array})."""
    work = Path(tempfile.mkdtemp(prefix="fkrige_"))
    try:
        pd.DataFrame({"row": tr, "s_x": S[tr, 0], "s_y": S[tr, 1], **columns}).to_csv(work / "training.csv", index=False)
        pd.DataFrame({"row": te, "s_x": S[te, 0], "s_y": S[te, 1]}).to_csv(work / "query.csv", index=False)
        if params is not None:
            (work / "params.json").write_text(json.dumps(params))
        subprocess.run(["Rscript", str(BLOCK_KRIGING_R), mode, str(work)], check=True, capture_output=True, timeout=1800)
        status = json.loads((work / "status.json").read_text()); out = {}
        for f in work.glob("*.csv"):
            if f.name not in ("training.csv", "query.csv"):
                out[f.stem] = np.loadtxt(f, delimiter=",", ndmin=1 if f.stem.endswith("_mean") else 2)
        for f in work.glob("fit_*.json"):
            out[f.stem] = json.loads(f.read_text())
        return status, out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def residual_columns(case):
    """Columns at the training rows for block_kriging.R: value (y), pred_ins (in-sample means) and pred_oof (cross-fitted means), from which
    R forms r_ins and r_oof exactly as kriging.R does, and z_oof (cross-fitted residuals standardised by the fold models'
    predictive standard deviation, computed here as in the WH implementation). Missing arrays are left out."""
    yt = case["y"][case["tr"]]; cols = {"value": yt}
    if case["m_ins"] is not None:
        cols["pred_ins"] = case["m_ins"]
    if case["m_oof"] is not None:
        cols["pred_oof"] = case["m_oof"]
        cols["z_oof"] = (yt - case["m_oof"]) / np.sqrt(np.clip(case["v_oof"], 1e-18, None))
    return cols


# ------------------------------------------------------------------------------------------------------------------------- blocks
def blocks_of(case):
    """Field and zone blocks (indices into te) and, per block, the revealed rows in the same area (for the sample mean)."""
    S, tr, te = case["S"], case["tr"], case["te"]
    blocks = zone_blocks(S, te)
    from sklearn.cluster import KMeans
    labels = KMeans(S_.ZONE_K, random_state=0, n_init=10).fit_predict(S)          # the clustering zone_blocks uses
    revealed = {b: (tr if b == "field" else tr[labels[tr] == int(b[4:])]) for b in blocks}
    return blocks, revealed


def case_rows(case, shapes=None, ours=True):
    """One row per block x method. shapes: {column: {'range_rel', 'nugget_share'}} for the cross-field variograms (None: local only).
    ours: also TabPFN-GP (J1pK, its kappa = 1 version J1p, and J1pL with the leave-one-field-out prior)."""
    S, tr, te, y = case["S"], case["tr"], case["te"], case["y"]
    n, yq = len(tr), y[te]; mT, sT = case["mT"], case["sT"]
    blocks, revealed = blocks_of(case)
    rows = []

    def add(variant, b, M, V, df=np.inf, failed=False, own_fallback=False, err=""):
        i = blocks[b]
        if not failed and not (np.isfinite(M) and np.isfinite(V) and V > 1e-18):      # a degenerate prediction is a failure
            failed, err = True, "non-finite mean or variance <= 1e-18"
        rows.append(dict(set=case["set"], src="external" if case["field"] in C.EXTERNAL else "limesoda", field=case["field"],
                         design=case["design"], rep=case["rep"], target=case["target"], n=n, m=len(te), block=b, mB=len(i),
                         nB=len(revealed[b]), variant=variant, M=float(M), V=float(V), df=float(df), obs=float(yq[i].mean()),
                         failed=bool(failed), own_fallback=bool(own_fallback), err=err))

    # TabPFN: independent and fully correlated errors
    for b, i in blocks.items():
        add("TAB", b, mT[i].mean(), np.sum(sT[i] ** 2) / len(i) ** 2)
        add("TAB1", b, mT[i].mean(), sT[i].mean() ** 2)
    # TabPFN-KpR, published settings and our TabPFN settings, combined as independent; a kriging failure is a failure
    for m in ("KpR", "KpR4"):
        if m in case:
            mK, sK, kf = case[m]
            for b, i in blocks.items():
                add(m, b, mK[i].mean(), np.sum(sK[i] ** 2) / len(i) ** 2, failed=kf, own_fallback=kf)
        else:
            for b in blocks:
                add(m, b, np.nan, np.nan, failed=True, err="missing output")
    # Kriging-based variants (block RK, the plug-in analogue, WH)
    cols = residual_columns(case)
    present = {"r_ins": "pred_ins" in cols, "r_oof": "pred_oof" in cols, "z_oof": "z_oof" in cols}
    params = {"shapes": {c: {"range": s["range_rel"] * field_extent(S), "nugget_share": s["nugget_share"]}
                         for c, s in (shapes or {}).items() if present.get(c)}}
    # Local (automap) and fixed (cross-field) variograms run in separate R processes, so that a crash of R in automap's local fit
    # (it segfaults on some n = 5 configurations) cannot take the fixed variants down with it.
    status, out, errs = {}, {}, {}
    for part in ("loc", "fix"):
        if len(cols) <= 1 or (part == "fix" and not params["shapes"]):
            continue
        try:
            st, o = run_r("all", S, tr, te, cols, {**params, "parts": [part]})
            status.update(st); out.update(o)
        except Exception as e:                                  # R itself failed: every variant of this part fails
            errs[part] = repr(e)[:150]
    err_of = lambda key: status.get(key, errs.get(key.split("_")[-1], "not computed"))   # noqa: E731
    for res, tag in (("rins", "ins"), ("roof", "oof")):
        for vg in ("loc", "fix"):
            key = f"{res}_{vg}"; good = status.get(key) == "ok"
            kmean, kcov = (out.get(f"{key}_mean"), out.get(f"{key}_cov")) if good else (None, None)
            for b, i in blocks.items():
                if good:
                    km, vk = float(kmean[i].mean()), float(kcov[np.ix_(i, i)].mean())
                    vT = float(np.sum(sT[i] ** 2)) / len(i) ** 2
                    add(f"RK_{tag}_{vg}", b, mT[i].mean() + km, vk)
                    add(f"RK_{tag}_{vg}_v", b, mT[i].mean() + km, vk + vT)
                    if tag == "oof":                            # plug-in analogue: kriging for the variance; TabPFN's mean below GATE_N
                        add(f"PI_{vg}", b, mT[i].mean() + (km if n >= F.GATE_N else 0.0), vk)
                else:
                    e = err_of(key)
                    for v in [f"RK_{tag}_{vg}", f"RK_{tag}_{vg}_v"] + ([f"PI_{vg}"] if tag == "oof" else []):
                        add(v, b, np.nan, np.nan, failed=True, err=str(e)[:150])
    for vg in ("loc", "fix"):
        key = f"z_{vg}"; good = status.get(key) == "ok"; R = out.get(f"corr_{key}") if good else None
        for b, i in blocks.items():
            if good:
                add(f"WH_{vg}", b, mT[i].mean(), float(sT[i] @ R[np.ix_(i, i)] @ sT[i]) / len(i) ** 2)
            else:
                add(f"WH_{vg}", b, np.nan, np.nan, failed=True, err=str(err_of(key))[:150])
    # Sample mean (design-based): location-scale t with n_B - 1 degrees of freedom, only where n_B >= 2
    for b, i in blocks.items():
        r = revealed[b]
        if len(r) >= 2:
            yr = y[r]; add("DB", b, yr.mean(), np.var(yr, ddof=1) * (1 / len(r) + 1 / len(i)), df=len(r) - 1)
    # TabPFN-GP: the block model J1p (population prior), J1pK = J1p with kappa = 1.10 (the factor setting), and the leave-one-field-out prior
    if ours:
        gate = n >= F.GATE_N
        for v, prior in (("J1p", PRIORS["POP"]), ("J1pL", PRIORS["LOFO"].get(case["field"], PRIORS["POP"]))):
            try:
                if case["m_oof"] is None:
                    raise ValueError("no cross-fitted residuals")
                res = block_model(prior, S, tr, te, y, case["ours"], blocks, sT, gate)
                for b, (M, V, mb) in res.items():
                    add(v, b, M, V)
                    if v == "J1p":
                        add("J1pK", b, M, F.KAPPA_OURS ** 2 * V)
            except Exception as e:
                for b in blocks:
                    add(v, b, np.nan, np.nan, failed=True, err=repr(e)[:150])
                    if v == "J1p":
                        add("J1pK", b, np.nan, np.nan, failed=True, err=repr(e)[:150])
    return rows


# ------------------------------------------------------------------------------------------------------------------------- shapes
def shape_fits(case):
    """Exponential variogram fits (nugget, psill, range) of r_ins, r_oof and z_oof for the cross-field shapes, with the field extent."""
    cols = residual_columns(case)
    status, out = run_r("fit", case["S"], case["tr"], case["te"], cols)
    L = field_extent(case["S"]); res = []
    for c in [c for c, p in (("r_ins", "pred_ins"), ("r_oof", "pred_oof"), ("z_oof", "z_oof")) if p in cols]:
        f = out.get(f"fit_{c}")
        if status.get(c) == "ok" and f is not None and (f["nugget"] + f["psill"]) > 0 and f["range"] > 0:
            res.append(dict(field=case["field"], design=case["design"], rep=case["rep"], target=case["target"], column=c,
                            range_rel=f["range"] / L, nugget_share=f["nugget"] / (f["nugget"] + f["psill"]), ok=True))
        else:
            res.append(dict(field=case["field"], design=case["design"], rep=case["rep"], target=case["target"], column=c, ok=False,
                            err=str(status.get(c))[:150]))
    return res
