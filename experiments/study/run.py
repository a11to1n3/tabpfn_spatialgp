"""The study of docs/PROTOCOL.md: data, predictions, area rows and the report (endpoints H1-H5 and the secondary analyses of Section 6).

  python -m experiments.study.run data --csv DIR
      <code>.npz in $SOILGP_DATA from the CSV files of export.R
  python -m experiments.study.run points [--fields F ...] [--reps R ...]
      TabPFN outputs (GPU; R for TabPFN-KpR) in $SOILGP_RUN/points_study
  python -m experiments.study.run rows --out DIR [--fields F ...] [--reps R ...]
      area rows of every method (pipeline/block_rows.py) and of the shared-error-only model (pipeline/further_models.py, variant O1)
  python -m experiments.study.run report --rows DIR --out DIR [--fields F ...]
      report.txt and the tables behind it

Methods (Section 2 of docs/PROTOCOL.md): TabPFN-GP is the block model J1p (correction of TabPFN's mean from n = 25,
population prior) scored with kappa = 1.15; the shared-error-only model is O1 (sigma^2 = 0) with kappa = 1.15; the comparators, their
factors and CRPS versions are those of configs/fixed/calibration.json. A field's properties are the columns of its CSV with a
value at every location. Designs n in {5, 10, 15, 20, 30, 50, 75, 100} where at least 10 locations remain, repeats 300-319.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one thread: TabPFN's output can depend on it
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import json                                                                   # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.paths import CONFIGS                                         # noqa: E402

CODES = ("SOIL250", "SOJA98", "DK_EST", "DK_FAA", "DK_JYN", "DK_SIL", "DK_VIN", "NI_BRA", "NI_MAR", "TR_CIH", "TR_KAD", "TR_KAR",
         "TR_KAT", "PO_NE", "PO_RS", "PO_PT", "PO_DG", "PO_MC", "PO_BR", "PO_RD", "PO_RT", "NX_ON", "NX_CZK", "NX_CZN", "NX_CZP", "NX_CZU",
         "NX_IT1", "NX_IT2", "NX_IT3", "NX_CAF", "NX_CHHB", "NX_CHZO", "DR_AUZ", "DR_EPO", "DR_PLO", "DR_CRO", "DR_F013", "DR_F033",
         "DR_F111", "DR_WAL", "DR_HEN", "DR_NWY", "DR_CAR")
REPS = range(300, 320)
KAPPA = 1.15
SMALL, LARGE = (5, 10), (30, 50, 75, 100)
BAND = (0.75, 0.85)
MARGIN = 0.10
NONINF = 1.02
KEYS = ["field", "design", "rep", "target", "block"]


def group(field):
    return ("Uniformity trials" if field in ("SOIL250", "SOJA98") else "SensRes" if field[:3] in ("DK_", "NI_", "TR_") else
            "Midwest" if field.startswith("PO_") else "New fields" if field.startswith("NX_") else "Further fields")


def evaluated(fields=None):
    """Fields with data (a field left without covariates or locations has no npz), in protocol order."""
    return tuple(f for f in (fields or CODES) if (F.C.DATA / f"{f}.npz").exists())


def tasks(field):
    Y = np.load(F.C.DATA / f"{field}.npz")["Y"]
    return tuple(t for j, t in enumerate(F.C.TARGETS) if np.isfinite(Y[:, j]).all())


def setup(fields):
    F.SETS["study"] = [(fields, F.C.N_DESIGNS, REPS)]
    F.POINTS["study"] = F.RUN / "points_study"
    F.targets = lambda field, set_name: tasks(field)
    from experiments.core import block_cases
    block_cases.FINAL_POINTS = F.POINTS["study"]
    return block_cases


# ------------------------------------------------------------------------------------------------------------------------- data
def stage_data(csv_dir):
    F.C.DATA.mkdir(parents=True, exist_ok=True)
    for f in CODES:
        p = Path(csv_dir) / f"{f}_points.csv"
        if not p.exists():
            print(f, "not evaluated (no CSV)", flush=True); continue
        d = pd.read_csv(p)
        cov = [c for c in d.columns if c not in ("x", "y", "SOC", "clay", "pH")]
        S = d[["x", "y"]].to_numpy(float); X = d[cov].to_numpy(float); Y = d[["SOC", "clay", "pH"]].to_numpy(float)
        assert len(np.unique(S, axis=0)) == len(S) and np.isfinite(X).all() and len(cov) >= 1 and len(S) >= 30
        assert all(np.isfinite(Y[:, j]).all() or np.isnan(Y[:, j]).all() for j in range(3)), f
        np.savez(F.C.DATA / f"{f}.npz", X=X, Y=Y, columns=np.array(cov), targets=np.array(F.C.TARGETS), folds=np.zeros(len(S), int),
                 coordinates=S)
        print(f, len(S), "locations; properties", tasks(f), ";", len(cov), "covariates", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- points
def stage_points(fields, reps):
    block_cases = setup(fields)
    from experiments.core.common import point_file, save_npz
    from experiments.core.point_model import kpr, our_model
    root = F.POINTS["study"]
    for field, data, design, rep, tr in block_cases.iter_splits("study", fields):
        if reps and rep not in reps:
            continue
        te = np.setdiff1d(np.arange(len(data["coordinates"])), tr)
        for target in F.targets(field, "study"):
            t0 = time.monotonic(); y = data["Y"][:, F.C.TARGETS.index(target)]
            split_arrays = dict(train_rows=tr, test_rows=te, observed=y[te])
            path = lambda m: point_file(root, m, design, field, target, rep)            # noqa: E731
            if not path("ours").exists():
                arrays, info = our_model(field, data, design, rep, tr, te, target, "ours", ("Gbma",))
                save_npz(path("ours"), **split_arrays, **arrays, info=json.dumps(info))
            if not path("KpR4").exists():                                         # TabPFN-KpR as selected (fixed/selection.json)
                res, failed = kpr(data, tr, te, target, ["KpR4"])
                meta = dict(field=field, target=target, design=design, rep=rep, model="KpR4", kriging_failed=failed)
                save_npz(path("KpR4"), **split_arrays, **res["KpR4"], metadata=json.dumps(meta))
            print(f"study {design} {field} {target} rep={rep} n={len(tr)} {time.monotonic() - t0:.1f} s", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- rows
def _rev_rows(job):
    from experiments.pipeline import further_models
    field, out, reps, fields = job
    setup(fields)
    further_models.FILTER.update({"designs": None, "reps": reps})
    return further_models.field_rows(("study", field, out, {"designs": None, "reps": reps}))


def stage_rows(fields, out, reps, workers):
    setup(fields)
    from experiments.pipeline import block_rows
    Path(out).mkdir(parents=True, exist_ok=True)
    shapes = json.loads((CONFIGS / "fixed" / "shapes.json").read_text())
    flt = {"designs": None, "reps": reps}
    with Pool(min(workers, len(fields))) as pool:
        for field, n, missing, sec in pool.imap_unordered(block_rows.field_rows, [("study", f, shapes, str(out), flt) for f in fields]):
            print(f"rows {field}: {n} rows, {len(missing)} cases without point outputs, {sec:.0f} s", flush=True)
        for field, n, missing, sec in pool.imap_unordered(_rev_rows, [(f, str(out), reps, fields) for f in fields]):
            print(f"shared-error-only rows {field}: {n} rows, {missing} cases without point outputs, {sec:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- report
def boot_idx(k, seed=F.SEED):
    return np.random.default_rng(seed).integers(0, k, (F.B, k))


def ratio_ci(num, den, idx):
    """Pooled ratio sum(num) / sum(den) over fields and its percentile bootstrap distribution over fields."""
    return float(num.sum() / den.sum()), num[idx].sum(1) / den[idx].sum(1)


def methods():
    cal = json.loads((CONFIGS / "fixed" / "calibration.json").read_text())["comparators"]
    comp = {cal[g]["variant"]: g for g in ("TabPFN", "TabPFN-KpR", "Block RK", "WH")}
    factors = {"J1p": KAPPA, "O1": KAPPA, **{v: cal[g]["factor"] for v, g in comp.items()}, "DB": cal["Sample mean"]["factor"]}
    crps_f = {**factors, **{cal[g]["variant"]: 1.0 for g in cal if cal[g]["crps_version"] != "calibrated"}}
    names = {"J1p": "TabPFN-GP", **comp, "DB": "Sample mean", "O1": "Shared-error-only"}
    return comp, factors, crps_f, names


def load_rows(rows_dir, fields):
    blocks = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(Path(rows_dir).glob("rows_study_*.csv"))], ignore_index=True)
    rev = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(Path(rows_dir).glob("models_study_*.csv"))], ignore_index=True)
    raw = pd.concat([blocks, rev[rev.variant == "O1"]], ignore_index=True)
    raw["failed"] = raw.failed.astype(bool)
    return raw[raw.field.isin(fields)]


def area_sets(raw, comp, factors, crps_f):
    """Scored rows: 'common' (TabPFN-GP and the four comparators all scored), the sample mean on its defined areas, and TabPFN-GP with
    the shared-error-only model on the areas where both are scored; coverage and width with the factors, CRPS with each method's version."""
    from experiments.core import scoring
    cov = scoring.score(raw, factors); crps = scoring.score(raw, crps_f)
    cov["crps"] = crps["crps"].to_numpy()
    five = ["J1p"] + list(comp)
    common = scoring.common_blocks(cov, five)
    common = common[common.variant.isin(five)]
    db = cov[(cov.variant == "DB") & cov.cov80.notna()]
    pair_o1 = scoring.common_blocks(cov, ["J1p", "O1"]); pair_o1 = pair_o1[pair_o1.variant.isin(["J1p", "O1"])]
    return cov, common, db, pair_o1


def kind_of(s):
    return np.where(s.block == "field", "field", "zone")


def coverage(s, v, kind, fields=None):
    """Pooled coverage of v over areas of one kind, with a 95 % percentile interval from resampling fields."""
    q = s[(s.variant == v) & (kind_of(s) == kind) & s.cov80.notna()]
    if fields is not None:
        q = q[q.field.isin(fields)]
    if q.empty:
        return np.nan, (np.nan, np.nan), 0
    g = q.groupby("field").cov80.agg(["sum", "count"]); est, b = ratio_ci(g["sum"].to_numpy(), g["count"].to_numpy(), boot_idx(len(g)))
    return est, tuple(np.percentile(b, [2.5, 97.5])), len(g)


def rise(s, v, kind):
    """D = coverage over areas with n in LARGE minus that with n in SMALL, over the fields with both groups; bootstrap distribution."""
    q = s[(s.variant == v) & (kind_of(s) == kind) & s.cov80.notna()]
    q = q.assign(grp=np.where(q.n.isin(SMALL), "S", np.where(q.n.isin(LARGE), "L", ""))); q = q[q.grp != ""]
    g = q.groupby(["field", "grp"]).cov80.agg(["sum", "count"]).unstack("grp").dropna()
    if g.empty:
        return np.nan, np.array([np.nan]), 0
    a = g.to_numpy(float); cols = [f"{x}_{y}" for x, y in g.columns]; c = {k: a[:, cols.index(k)] for k in cols}
    idx = boot_idx(len(g))
    est = c["sum_L"].sum() / c["count_L"].sum() - c["sum_S"].sum() / c["count_S"].sum()
    b = c["sum_L"][idx].sum(1) / c["count_L"][idx].sum(1) - c["sum_S"][idx].sum(1) / c["count_S"][idx].sum(1)
    return float(est), b, len(g)


def crps_contrast(s, a, b, kind, col="crps", fields=None):
    """Ratio a / b: per task the log ratio of summed values over the areas where both are scored, mean over a field's properties,
    geometric mean over fields, 95 % percentile interval from resampling fields; field values for the Wilcoxon test."""
    from experiments.core import scoring
    return scoring.contrast(s, a, b, kind, col=col, fields=fields)


def stage_report(rows_dir, out, fields):
    from experiments.core import scoring
    comp, factors, crps_f, names = methods()
    raw = load_rows(rows_dir, fields)
    cov, common, db, pair_o1 = area_sets(raw, comp, factors, crps_f)
    main = pd.concat([common, db], ignore_index=True)
    lines = []

    def say(s=""):
        lines.append(s); print(s, flush=True)

    fmt = lambda e, ci: f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"                  # noqa: E731
    say("=== Field and compact-zone means: TabPFN-GP intervals (PROTOCOL.md) ===")
    say(f"fields evaluated: {raw.field.nunique()} of {len(CODES)}; tasks (field x property): {raw.groupby(['field', 'target']).ngroups}; "
        f"cases: {raw[raw.variant == 'J1p'].groupby(['field', 'design', 'rep', 'target']).ngroups}")
    say(f"areas scored by all methods: {(common[common.variant == 'J1p'].block == 'field').sum()} field and "
        f"{(common[common.variant == 'J1p'].block != 'field').sum()} zone means; sample mean: {(db.block == 'field').sum()} field and "
        f"{(db.block != 'field').sum()} zone means; TabPFN-GP with the shared-error-only model: "
        f"{(pair_o1[pair_o1.variant == 'O1'].block == 'field').sum()} field and {(pair_o1[pair_o1.variant == 'O1'].block != 'field').sum()} zone means")

    say("\n-- endpoints --")
    # H1
    h1 = {}
    for v in comp:
        est, b, k = rise(common, v, "field"); h1[v] = np.percentile(b, 2.5)
        say(f"H1 {names[v]:12s} field-mean D = {est:+.3f}, 95 % [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] ({k} fields)")
    say(f"H1 (coverage rises with n without a shared-error term; all four lower bounds above 0): {'met' if all(x > 0 for x in h1.values()) else 'not met'}")
    # H2
    for kind in ("field", "zone"):
        est, b, k = rise(common, "J1p", kind); lo, hi = np.percentile(b, [5, 95])
        say(f"H2 TabPFN-GP {kind}-mean D = {est:+.3f}, 90 % [{lo:+.3f}, {hi:+.3f}] ({k} fields): "
            f"{'met' if -MARGIN <= lo and hi <= MARGIN else 'not met'}")
    # H3
    for kind in ("field", "zone"):
        est, ci, k = coverage(common, "J1p", kind)
        say(f"H3 TabPFN-GP {kind}-mean coverage {fmt(est, ci)} ({k} fields): {'met' if BAND[0] <= ci[0] and ci[1] <= BAND[1] else 'not met'}")
    # H4
    est, ci, k = coverage(pair_o1, "O1", "field"); c = crps_contrast(pair_o1, "O1", "J1p", "field")
    ok_a = c is not None and BAND[0] <= ci[0] and ci[1] <= BAND[1] and c["ci"][1] < NONINF
    say(f"H4(a) shared-error-only field-mean coverage {fmt(est, ci)}; CRPS ratio to TabPFN-GP {fmt(c['ratio'], c['ci']) if c else '-'} ({k} fields): "
        f"{'met' if ok_a else 'not met'}")
    z = pair_o1[(kind_of(pair_o1) == "zone")].pivot_table(index=KEYS, columns="variant", values="cov80").dropna().reset_index()
    g = z.groupby("field").agg(gp=("J1p", "sum"), o1=("O1", "sum"), m=("J1p", "size")); idx = boot_idx(len(g))
    a, o, m = (g[x].to_numpy(float) for x in ("gp", "o1", "m"))
    d_est = (a.sum() - o.sum()) / m.sum(); d_b = (a[idx].sum(1) - o[idx].sum(1)) / m[idx].sum(1)
    say(f"H4(b) TabPFN-GP minus shared-error-only zone-mean coverage {d_est:+.3f}, 95 % [{np.percentile(d_b, 2.5):+.3f}, "
        f"{np.percentile(d_b, 97.5):+.3f}] ({len(g)} fields): {'met' if np.percentile(d_b, 2.5) > 0 else 'not met'}")
    # H5
    res = []
    for kind in ("field", "zone"):
        for v in list(comp) + ["DB"]:
            c = crps_contrast(common if v != "DB" else pd.concat([common[common.variant == "J1p"], db]), "J1p", v, kind)
            res.append((kind, v, c, scoring.wilcoxon_less(c["Lf"]) if c else np.nan))
    adj = scoring.holm([r[3] for r in res])
    for (kind, v, c, p), pa in zip(res, adj):
        if c is None:
            say(f"H5 {kind:5s} TabPFN-GP / {names[v]:12s} no areas scored by both"); continue
        verdict = ("better" if pa < 0.05 and c["ratio"] < 1 else "no worse" if c["ci"][1] < NONINF else "neither better nor shown no worse")
        say(f"H5 {kind:5s} TabPFN-GP / {names[v]:12s} CRPS ratio {fmt(c['ratio'], c['ci'])}, lower on {c['better']} of {c['fields']} fields, "
            f"Wilcoxon p = {p:.4g}, Holm {pa:.4g}: {verdict}")

    say("\n-- secondary analyses --")
    allv = ["J1p"] + list(comp) + ["DB", "O1"]
    src = lambda v: pair_o1 if v == "O1" else main                            # noqa: E731
    fig = []
    ns = sorted(raw.n.unique())
    say("coverage by n [95 % interval over fields]:")
    for kind in ("field", "zone"):
        for v in allv:
            parts = []
            for n in ns:
                est, ci, k = coverage(src(v)[src(v).n == n], v, kind)
                if np.isfinite(est):
                    parts.append(f"{n}: {fmt(est, ci)}"); fig.append(dict(kind=kind, method=names[v], n=int(n), coverage=est, lo=ci[0], hi=ci[1], fields=k))
            say(f"  {kind:5s} {names[v]:17s} " + "; ".join(parts))
    say("rise in coverage D (n in 30-100 minus n in 5-10) [95 % interval]:")
    for kind, vs in (("zone", list(comp)), ("field", ["DB"]), ("zone", ["DB"])):
        for v in vs:
            est, b, k = rise(main, v, kind)
            say(f"  {kind:5s} {names[v]:12s} D = {est:+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] ({k} fields)")
    say("by group of fields: coverage (field | zone) and CRPS ratio TabPFN-GP / method (field | zone) [95 % intervals over the group's fields]:")
    for gname in ("Uniformity trials", "SensRes", "Midwest", "New fields", "Further fields"):
        gf = [f for f in fields if group(f) == gname and f in set(raw.field)]
        if not gf:
            continue
        say(f"  {gname} ({len(gf)} fields)")
        for v in allv:
            cv = [coverage(src(v), v, kind, gf) for kind in ("field", "zone")]
            txt = f"    {names[v]:17s} coverage {fmt(*cv[0][:2])} | {fmt(*cv[1][:2])}"
            if v not in ("J1p", "O1"):
                cc = [crps_contrast(common if v != "DB" else pd.concat([common[common.variant == "J1p"], db]), "J1p", v, kind, fields=gf)
                      for kind in ("field", "zone")]
                txt += "; CRPS ratio " + " | ".join(fmt(c["ratio"], c["ci"]) if c else "-" for c in cc)
            say(txt)
    say("interval-width ratio TabPFN-GP / method (80 % intervals with the factors) [95 % interval over fields]:")
    for kind in ("field", "zone"):
        parts = []
        for v in list(comp) + ["DB", "O1"]:
            s = pair_o1 if v == "O1" else common if v != "DB" else pd.concat([common[common.variant == "J1p"], db])
            c = crps_contrast(s, "J1p", v, kind, col="width"); parts.append(f"{names[v]} {fmt(c['ratio'], c['ci']) if c else '-'}")
        say(f"  {kind:5s} " + "; ".join(parts))
    say("coverage in zones without a sample [95 % interval]: " + "; ".join(
        f"{names[v]} {fmt(*coverage(q[q.nB == 0], v, 'zone')[:2])}" for v in ["J1p"] + list(comp) + ["O1"] for q in [src(v)]))
    say("coverage by field (field | zone):")
    for f in fields:
        if f not in set(raw.field):
            continue
        say(f"  {f:8s} " + "; ".join(f"{names[v]} {coverage(src(v), v, 'field', [f])[0]:.3f} | {coverage(src(v), v, 'zone', [f])[0]:.3f}"
                                     for v in allv))
    say("coverage by property (field | zone) [95 % interval over fields]:")
    for t in F.C.TARGETS:
        parts = []
        for v in allv:
            q = src(v)[src(v).target == t]
            if q.empty:
                continue
            a, b = coverage(q, v, "field"), coverage(q, v, "zone")
            parts.append(f"{names[v]} {fmt(*a[:2])} | {fmt(*b[:2])}")
        if parts:
            say(f"  {t.replace('_target', ''):5s} ({src('J1p')[src('J1p').target == t].field.nunique()} fields) " + "; ".join(parts))
    say("failed predictions by method and n (failed / areas):")
    for v in ["J1p"] + list(comp) + ["DB", "O1"]:
        q = raw[raw.variant == v]
        say(f"  {names[v]:17s} " + "; ".join(f"{n}: {int(q[q.n == n].failed.sum())}/{int((q.n == n).sum())}" for n in ns))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(fig).to_csv(out / "coverage_by_n.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["data", "points", "rows", "report"])
    ap.add_argument("--csv"); ap.add_argument("--rows"); ap.add_argument("--out"); ap.add_argument("--fields", nargs="*")
    ap.add_argument("--reps", nargs="*", type=int, help="dry run only"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "data":
        return stage_data(a.csv)
    fields = tuple(a.fields) if a.fields else evaluated()
    if a.stage == "points":
        stage_points(fields, a.reps)
    elif a.stage == "rows":
        stage_rows(fields, a.out, a.reps, a.workers)
    else:
        stage_report(a.rows, a.out, fields)


if __name__ == "__main__":
    main()
