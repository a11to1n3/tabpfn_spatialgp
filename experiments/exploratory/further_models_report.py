"""E9-E11 of docs/exploratory/plan_E9-E13.md (CPU): factors by one rule, further models, and inference by dataset and group.

  python -m experiments.exploratory.further_models_report <work>/rows <work>/variants <repro>/work/factor_setting <work>/report/field_areas.csv <out>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, t as tdist

ROWS, VAR, FIRST, AREAS, OUT = (Path(a) for a in sys.argv[1:6])
from experiments.study import run as S                                        # noqa: E402
from experiments.core import scoring                                          # noqa: E402

F = S.F
GRID = np.round(np.arange(0.50, 5.0001, 0.05), 2)
lines, csv_rows = [], []


def say(s=""):
    lines.append(s); print(s, flush=True)


def fmt(e, ci):
    return f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


DATASET = {"SOIL250": "SOIL250", "SOJA98": "SOJA98", "NX_ON": "Ontario", "NX_CAF": "Cook farm", "DR_AUZ": "Auzeville", "DR_EPO": "Epoisses",
           "DR_PLO": "Ploudaniel", "DR_CRO": "Crouel", "DR_WAL": "Walnut Creek", "DR_CAR": "Caroline"}
for f in S.CODES:
    if f[:3] in ("DK_", "NI_", "TR_"):
        DATASET[f] = "SensRes"
    elif f.startswith("PO_"):
        DATASET[f] = "Midwest"
    elif f.startswith("NX_CZ"):
        DATASET[f] = "Czech STEROPES"
    elif f.startswith("NX_IT"):
        DATASET[f] = "Italian STEROPES"
    elif f.startswith("NX_CH"):
        DATASET[f] = "Agroscope"
    elif f in ("DR_F013", "DR_F033", "DR_F111"):
        DATASET[f] = "Maricopa"
    elif f in ("DR_HEN", "DR_NWY"):
        DATASET[f] = "EIDC"
PUBLIC = {"NI_MAR", "NX_CZK", "NX_CZN", "NX_CZP", "NX_CZU", "NX_IT1", "NX_IT2", "NX_IT3", "NX_CHHB", "NX_CHZO", "DR_AUZ", "DR_EPO", "DR_PLO",
          "DR_CRO", "DR_F013", "DR_F111", "DR_WAL", "DR_HEN"} | {f for f in S.CODES if f.startswith("PO_")}
GROUPS = ("Uniformity trials", "SensRes", "Midwest", "New fields", "Further fields")


def once(d):
    return d[~((d.field == "OXFNC") | ((d.field == "MEUSE") & (d.target != "SOC_target")))]


def kind_of(s):
    return np.where(s.block == "field", "field", "zone")


# ------------------------------------------------------------------------------------------------------------------- factors (E10)
def factor_rule(r, v):
    """The paper's rule, each external task once: smallest grid factor whose mean of pooled field and zone 80 % coverage reaches 0.80."""
    s = once(r[(r.variant == v) & ~r.failed.astype(bool)])
    sd = np.sqrt(np.clip(s.V.to_numpy(), 0, None)); df = s.df.to_numpy(); err = np.abs(s.obs.to_numpy() - s.M.to_numpy())
    q = np.where(np.isfinite(df), tdist.ppf(0.9, np.where(np.isfinite(df), df, 2.0)), norm.ppf(0.9)); fld = (s.block == "field").to_numpy()
    for f in GRID:
        c = err <= q * f * sd
        a = c[fld].mean() if fld.any() else np.nan; b = c[~fld].mean() if (~fld).any() else np.nan
        val = np.nanmean([a, b])
        if val >= 0.80:
            return float(f), float(val)
    return float(GRID[-1]), float(val)


def crps_version(r, v, f):
    lim = r[(r.src == "limesoda") & (r.variant == v)]
    both = pd.concat([lim, lim.assign(variant=v + "__cal")], ignore_index=True)
    sc = scoring.score(both, {v + "__cal": f}); ks = [k for k in ("field", "zone") if (scoring.kind_rows(sc, k).variant == v).any()]
    ratio = float(np.exp(np.mean([np.log(scoring.contrast(sc, v + "__cal", v, k)["ratio"]) for k in ks])))
    return "calibrated" if ratio < 1 else "uncalibrated", ratio


def main():
    global csv_rows
    OUT.mkdir(parents=True, exist_ok=True)
    say("=== E9-E11 (docs/exploratory/plan_E9-E13.md; specified after the study's results had been seen; exploratory) ===")
    first = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(FIRST.glob("rows_first_*.csv"))] +
                      [pd.read_csv(p, low_memory=False) for p in sorted(VAR.glob("variants_first_*.csv"))], ignore_index=True)
    first["failed"] = first.failed.astype(bool)
    models = {"TabPFN-GP": "J1p", "TabPFN": "TAB", "TabPFN-KpR": "KpR4", "Block RK": "RK_ins_fix_v", "WH": "WH_fix", "Sample mean": "DB",
              "Shared-error-only": "O1", "Difference estimator": "DE", "Difference estimator, zones": "DEz",
              "Block RK, held-out errors": "RK_oof_fix_v", "WH, local correlogram": "WH_loc", "TabPFN-KpR, published settings": "KpR",
              "Zone-effect model": "Z", "TabPFN-GP, prior without dummy coordinates": "GPx", "TabPFN-GP, no prior": "GPnp",
              "TabPFN-GP, no averaging": "GPna", "TabPFN-GP, neither": "GPnn"}
    say("\n-- E10: factors by one rule (each external task once) and CRPS version --")
    fac, crps_f = {}, {}
    for name, v in models.items():
        f, cov = factor_rule(first, v); ver, ratio = crps_version(first, v, f)
        fac[v] = f; crps_f[v] = f if ver == "calibrated" else 1.0
        say(f"  {name:45s} {v:13s} factor {f:.2f} (coverage {cov:.3f}); CRPS version {ver} ({ratio:.3f})")
        csv_rows.append(dict(analysis="E10", what="factor", model=name, variant=v, est=f, lo=cov, hi=ratio, n=np.nan))
    (OUT / "factors_once.json").write_text(json.dumps({"factor": fac, "crps_factor": crps_f}, indent=1))

    # ---------------------------------------------------------------------------------------------------- study rows
    raw = S.load_rows(ROWS, S.CODES)
    var = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(VAR.glob("variants_study_*.csv"))], ignore_index=True)
    var["failed"] = var.failed.astype(bool)
    comp_once = {"TAB": "TabPFN", "KpR4": "TabPFN-KpR", "RK_ins_fix_v": "Block RK", "WH_fix": "WH"}
    factors_once = {"J1p": S.KAPPA, "O1": fac["O1"], **{v: fac[v] for v in comp_once}, "DB": fac["DB"]}
    crps_once = {**factors_once, **{v: crps_f[v] for v in list(comp_once) + ["DB", "O1"]}}
    crps_once["J1p"] = S.KAPPA

    # E10: the report with these factors (the study's own report code)
    orig_methods = S.methods
    S.methods = lambda: (comp_once, factors_once, crps_once, {"J1p": "TabPFN-GP", **comp_once, "DB": "Sample mean", "O1": "Shared-error-only"})
    say("\n-- E10: the study's report with these factors (TabPFN-GP kappa = 1.15) --> report_once/report.txt --")
    S.stage_report(str(ROWS), str(OUT / "report_once"), S.evaluated())
    say("\n-- E10b: the same, with TabPFN-GP's CRPS also by the version rule (without its factor) --> report_once_gp1/report.txt --")
    crps_gp1 = dict(crps_once, J1p=1.0)
    S.methods = lambda: (comp_once, factors_once, crps_gp1, {"J1p": "TabPFN-GP", **comp_once, "DB": "Sample mean", "O1": "Shared-error-only"})
    S.stage_report(str(ROWS), str(OUT / "report_once_gp1"), S.evaluated())
    S.methods = orig_methods
    comp, factors, crps_fac, names = S.methods()
    cov, common, db, pair_o1 = S.area_sets(raw, comp, factors, crps_fac)
    c = S.crps_contrast(pair_o1, "J1p", "O1", "zone"); say(f"\n  zone CRPS ratio TabPFN-GP / shared-error-only (kappa 1.15 for both): {fmt(c['ratio'], c['ci'])}")
    sc_o1 = scoring.score(pair_o1.drop(columns=["crps", "cov80", "width"]), {"J1p": S.KAPPA, "O1": fac["O1"]})
    c = S.crps_contrast(sc_o1, "J1p", "O1", "zone"); say(f"  zone CRPS ratio TabPFN-GP / shared-error-only with its own factor {fac['O1']:.2f}: {fmt(c['ratio'], c['ci'])}")
    for k in ("field", "zone"):
        e, ci, n = S.coverage(sc_o1.assign(cov80=sc_o1.cov80), "O1", k); say(f"  shared-error-only coverage with its own factor, {k}: {fmt(e, ci)}")

    # ------------------------------------------------------------------------------------------------ E11 further models
    say("\n-- E11: further models on the 43 fields, each with its own factor (rule above); CRPS of both by the version rule --")
    gp = raw[raw.variant == "J1p"]
    for name, v in models.items():
        if v in ("J1p", "TAB", "KpR4", "RK_ins_fix_v", "WH_fix", "DB", "O1", "KpR"):          # KpR (published settings): not run in the study
            continue
        src = var if v in ("Z", "GPx", "GPnp", "GPna", "GPnn", "DE", "DEz") else raw
        rows = pd.concat([gp, src[src.variant == v]], ignore_index=True)
        sc_c = scoring.score(rows, {"J1p": S.KAPPA, v: fac[v]}); sc_r = scoring.score(rows, {"J1p": crps_f["J1p"], v: crps_f[v]})   # CRPS: rule version for both
        sc_c["crps"] = sc_r["crps"].to_numpy()
        both = scoring.common_blocks(sc_c, ["J1p", v]); both = both[both.variant.isin(["J1p", v])]
        parts = [f"{name} ({v}, factor {fac[v]:.2f})"]
        for k in ("field", "zone"):
            if not (S.kind_of(both[both.variant == v]) == k).any():
                continue
            e, ci, n = S.coverage(both, v, k); est, b, nf = S.rise(both, v, k)
            cc = S.crps_contrast(both, "J1p", v, k)
            parts.append(f"{k}: coverage {fmt(e, ci)}, D {est:+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}], "
                         f"CRPS ratio TabPFN-GP / model {fmt(cc['ratio'], cc['ci'])} ({cc['better']}/{cc['fields']} fields lower)")
            csv_rows += [dict(analysis="E11", what="coverage", model=name, variant=v, kind=k, est=e, lo=ci[0], hi=ci[1], n=n),
                         dict(analysis="E11", what="D", model=name, variant=v, kind=k, est=est, lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), n=nf),
                         dict(analysis="E11", what="crps_ratio", model=name, variant=v, kind=k, est=cc["ratio"], lo=cc["ci"][0], hi=cc["ci"][1], n=cc["fields"])]
        say("  " + "\n      ".join(parts))

    # ------------------------------------------------------------------------------------------------ E9 inference
    say("\n-- E9.1: the five tests with bootstrap over the 17 source datasets --")
    main_set = pd.concat([common, db], ignore_index=True)
    ds = sorted(set(DATASET[f] for f in S.evaluated())); say(f"  datasets: {len(ds)}")
    rng = np.random.default_rng(F.SEED); idx = rng.integers(0, len(ds), (F.B, len(ds)))

    def by_ds(series):
        return series.groupby(series.index.map(DATASET)).sum().reindex(ds).fillna(0).to_numpy(float)

    def cov_ds(s, v, k):
        q = s[(s.variant == v) & (S.kind_of(s) == k) & s.cov80.notna()]; g = q.groupby("field").cov80.agg(["sum", "count"])
        a, n = by_ds(g["sum"]), by_ds(g["count"]); b = a[idx].sum(1) / n[idx].sum(1)
        return a.sum() / n.sum(), np.percentile(b, [2.5, 97.5])

    def rise_ds(s, v, k, level=95):
        q = s[(s.variant == v) & (S.kind_of(s) == k) & s.cov80.notna()]
        q = q.assign(grp=np.where(q.n.isin(S.SMALL), "S", np.where(q.n.isin(S.LARGE), "L", ""))); q = q[q.grp != ""]
        g = q.groupby(["field", "grp"]).cov80.agg(["sum", "count"]).unstack("grp").dropna()
        cols = {f"{x}_{y}": g[(x, y)] for x, y in g.columns}; A = {k2: by_ds(v2) for k2, v2 in cols.items()}
        est = A["sum_L"].sum() / A["count_L"].sum() - A["sum_S"].sum() / A["count_S"].sum()
        b = A["sum_L"][idx].sum(1) / A["count_L"][idx].sum(1) - A["sum_S"][idx].sum(1) / A["count_S"][idx].sum(1)
        lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
        return est, np.nanpercentile(b, [lo, hi])

    for v, name in comp.items():
        e, ci = rise_ds(main_set, v, "field"); say(f"  test 1 {name:12s} D {e:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        csv_rows.append(dict(analysis="E9.1", what="t1", model=name, kind="field", est=e, lo=ci[0], hi=ci[1]))
    for k in ("field", "zone"):
        e, ci = rise_ds(main_set, "J1p", k, 90); say(f"  test 2 TabPFN-GP {k} D {e:+.3f}, 90 % [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        csv_rows.append(dict(analysis="E9.1", what="t2", model="TabPFN-GP", kind=k, est=e, lo=ci[0], hi=ci[1]))
    for k in ("field", "zone"):
        e, ci = cov_ds(main_set, "J1p", k); say(f"  test 3 TabPFN-GP {k} coverage {e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        csv_rows.append(dict(analysis="E9.1", what="t3", model="TabPFN-GP", kind=k, est=e, lo=ci[0], hi=ci[1]))
    e, ci = cov_ds(pair_o1, "O1", "field"); say(f"  test 4a shared-error-only field coverage {e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    csv_rows.append(dict(analysis="E9.1", what="t4a_cov", model="Shared-error-only", kind="field", est=e, lo=ci[0], hi=ci[1]))
    q = pair_o1[(S.kind_of(pair_o1) == "zone") & pair_o1.cov80.notna()]
    g1 = q[q.variant == "J1p"].groupby("field").cov80.agg(["sum", "count"]); g0 = q[q.variant == "O1"].groupby("field").cov80.agg(["sum", "count"])
    a1, n1, a0, n0 = by_ds(g1["sum"]), by_ds(g1["count"]), by_ds(g0["sum"]), by_ds(g0["count"])
    b = a1[idx].sum(1) / n1[idx].sum(1) - a0[idx].sum(1) / n0[idx].sum(1)
    say(f"  test 4b zone coverage difference {a1.sum() / n1.sum() - a0.sum() / n0.sum():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
    csv_rows.append(dict(analysis="E9.1", what="t4b", model="TabPFN-GP", kind="zone", est=a1.sum() / n1.sum() - a0.sum() / n0.sum(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
    pv, labs, ests = [], [], []
    cc = S.crps_contrast(pair_o1, "O1", "J1p", "field"); Ld = cc["Lf"].groupby(cc["Lf"].index.map(DATASET)).mean()
    bb = np.exp(Ld.to_numpy()[np.random.default_rng(F.SEED).integers(0, len(Ld), (F.B, len(Ld)))].mean(1))
    say(f"  test 4a CRPS ratio shared-error-only / TabPFN-GP {np.exp(Ld.mean()):.3f} [{np.percentile(bb, 2.5):.3f}, {np.percentile(bb, 97.5):.3f}] ({len(Ld)} datasets)")
    csv_rows.append(dict(analysis="E9.1", what="t4a_crps", model="Shared-error-only", kind="field", est=np.exp(Ld.mean()), lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5)))
    for k in ("field", "zone"):
        for v in list(comp) + ["DB"]:
            s = common if v != "DB" else pd.concat([common[common.variant == "J1p"], db])
            cc = S.crps_contrast(s, "J1p", v, k); Ld = cc["Lf"].groupby(cc["Lf"].index.map(DATASET)).mean()
            bb = np.exp(Ld.to_numpy()[np.random.default_rng(F.SEED).integers(0, len(Ld), (F.B, len(Ld)))].mean(1))
            pv.append(scoring.wilcoxon_less(Ld.to_numpy())); labs.append(f"{k} {names[v]}"); ests.append((np.exp(Ld.mean()), np.percentile(bb, [2.5, 97.5]), int((Ld < 0).sum()), len(Ld)))
    for lab, (e, ci, lo, nd), p, ph in zip(labs, ests, pv, scoring.holm(pv)):
        say(f"  test 5 {lab:26s} ratio {e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], lower in {lo}/{nd} datasets, Wilcoxon p {p:.3g}, Holm {ph:.3g}")
        csv_rows.append(dict(analysis="E9.1", what="t5", model=lab.split(" ", 1)[1], kind=lab.split(" ")[0], est=e, lo=ci[0], hi=ci[1], n=ph))

    say("\n-- E9.2: leave-one-group-out (each criterion: number of the five omissions in which it is met; estimate range) --")
    res = {}
    for g in GROUPS:
        keep = [f for f in S.evaluated() if S.group(f) != g]
        ms = main_set[main_set.field.isin(keep)]; po = pair_o1[pair_o1.field.isin(keep)]; cm = common[common.field.isin(keep)]
        r = {}
        for v, name in comp.items():
            est, b, k = S.rise(ms, v, "field"); r[f"t1 {name}"] = (est, np.percentile(b, 2.5) > 0)
        for k in ("field", "zone"):
            est, b, n = S.rise(ms, "J1p", k); lo, hi = np.percentile(b, [5, 95]); r[f"t2 {k}"] = (est, lo > -0.10 and hi < 0.10)
            e, ci, n = S.coverage(ms, "J1p", k); r[f"t3 {k}"] = (e, ci[0] >= 0.75 and ci[1] <= 0.85)
        e, ci, n = S.coverage(po, "O1", "field"); cc = S.crps_contrast(po, "O1", "J1p", "field")
        r["t4a"] = (e, ci[0] >= 0.75 and ci[1] <= 0.85 and cc["ci"][1] < 1.02)
        q = po[(S.kind_of(po) == "zone") & po.cov80.notna()]
        g1 = q[q.variant == "J1p"].groupby("field").cov80.agg(["sum", "count"]); g0 = q[q.variant == "O1"].groupby("field").cov80.agg(["sum", "count"])
        ii = S.boot_idx(len(g1)); bd = g1["sum"].to_numpy()[ii].sum(1) / g1["count"].to_numpy()[ii].sum(1) - g0["sum"].to_numpy()[ii].sum(1) / g0["count"].to_numpy()[ii].sum(1)
        r["t4b"] = (g1["sum"].sum() / g1["count"].sum() - g0["sum"].sum() / g0["count"].sum(), np.percentile(bd, 2.5) > 0)
        pp, ll = [], []
        for k in ("field", "zone"):
            for v in list(comp) + ["DB"]:
                s = cm if v != "DB" else pd.concat([cm[cm.variant == "J1p"], db[db.field.isin(keep)]])
                cc = S.crps_contrast(s, "J1p", v, k); pp.append(scoring.wilcoxon_less(cc["Lf"].to_numpy())); ll.append((f"t5 {k} {names[v]}", cc))
        for (lab, cc), ph in zip(ll, scoring.holm(pp)):
            r[lab] = (cc["ratio"], (ph < 0.05 and cc["ratio"] < 1) or cc["ci"][1] < 1.02)
        res[g] = r
    for key in res[GROUPS[0]]:
        vals = [res[g][key][0] for g in GROUPS]; met = sum(bool(res[g][key][1]) for g in GROUPS)
        say(f"  {key:28s} met in {met}/5; estimate {min(vals):+.3f} to {max(vals):+.3f}")
        csv_rows.append(dict(analysis="E9.2", what=key, est=met, lo=min(vals), hi=max(vals)))

    say("\n-- E9.3: per-group test quantities --")
    for g in GROUPS:
        fl = [f for f in S.evaluated() if S.group(f) == g]; ms = main_set[main_set.field.isin(fl)]; po = pair_o1[pair_o1.field.isin(fl)]
        parts = []
        for k in ("field", "zone"):
            est, b, n = S.rise(ms, "J1p", k); lo, hi = np.percentile(b, [5, 95]); parts.append(f"t2 {k} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for k in ("field", "zone"):
            e, ci, n = S.coverage(ms, "J1p", k); parts.append(f"t3 {k} {fmt(e, ci)}")
        e, ci, n = S.coverage(po, "O1", "field"); parts.append(f"t4a {fmt(e, ci)}")
        say(f"  {g} ({len(fl)} fields): " + "; ".join(parts))

    say("\n-- E9.4: field-balanced rise D (fields with n = 5, 10, 30 and 50; equal weight per field) --")
    for v, name in list({"J1p": "TabPFN-GP", **comp, "DB": "Sample mean"}.items()):
        for k in ("field", "zone"):
            q = main_set[(main_set.variant == v) & (S.kind_of(main_set) == k) & main_set.cov80.notna() & main_set.n.isin([5, 10, 30, 50])]
            per = q.groupby(["field", "n"]).cov80.mean().unstack("n")
            per = per.dropna(subset=[5, 10, 30, 50]) if all(c in per.columns for c in (5, 10, 30, 50)) else per.iloc[0:0]
            d = ((per[30] + per[50]) / 2 - (per[5] + per[10]) / 2).to_numpy()
            b = d[S.boot_idx(len(d))].mean(1)
            say(f"  {name:12s} {k}: D {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] ({len(d)} fields)")

    say("\n-- E9.5: TabPFN-GP coverage by size --")
    ar = pd.read_csv(AREAS).set_index("field"); gpm = main_set[main_set.variant == "J1p"].copy()
    gpm["N"] = gpm.field.map(ar.locations); gpm["area"] = gpm.field.map(ar.area_ha)
    t1, t2 = np.quantile(ar.loc[list(S.evaluated()), "locations"], [1 / 3, 2 / 3])
    order = [f"N <= {t1:.0f}", f"{t1:.0f} < N <= {t2:.0f}", f"N > {t2:.0f}"]
    gpm["Ncls"] = np.where(gpm.N <= t1, order[0], np.where(gpm.N <= t2, order[1], order[2]))
    for k in ("field", "zone"):
        q = gpm[S.kind_of(gpm) == k]
        say(f"  {k}, by N tertile: " + "; ".join(f"{c}: {fmt(*S.coverage(q[q.Ncls == c], 'J1p', k)[:2])} ({q[q.Ncls == c].field.nunique()} fields)"
                                                  for c in order))
        per = q.groupby("field").cov80.mean(); rho = spearmanr(per.index.map(ar.locations), per.to_numpy())
        say(f"     Spearman correlation of per-field coverage with N: {rho.statistic:+.2f} (p = {rho.pvalue:.3g})")
        qa = q[~q.field.isin(["SOIL250", "SOJA98"])]
        acls = pd.cut(qa.area, [0, 1, 10, 100, np.inf], labels=["< 1 ha", "1-10 ha", "10-100 ha", "> 100 ha"])
        say(f"     by area: " + "; ".join(f"{c}: {fmt(*S.coverage(qa[acls == c], 'J1p', k)[:2])} ({qa[acls == c].field.nunique()})" for c in acls.cat.categories))
    q = gpm[S.kind_of(gpm) == "zone"]; mcls = pd.cut(q.mB, [2, 9, 29, np.inf], labels=["3-9", "10-29", ">= 30"])
    say(f"  zones by m_B: " + "; ".join(f"{c}: {fmt(*S.coverage(q[mcls == c], 'J1p', 'zone')[:2])}" for c in mcls.cat.categories))

    say("\n-- E9.6: by covariate source (released covariates vs public only) --")
    for lab, fl in (("released", [f for f in S.evaluated() if f not in PUBLIC]), ("public", [f for f in S.evaluated() if f in PUBLIC])):
        parts = []
        for k in ("field", "zone"):
            e, ci, n = S.coverage(main_set, "J1p", k, fl)
            c1 = S.crps_contrast(pd.concat([common[common.variant == "J1p"], db]), "J1p", "DB", k, fields=fl)
            c2 = S.crps_contrast(common, "J1p", "TAB", k, fields=fl)
            parts.append(f"{k}: coverage {fmt(e, ci)}, CRPS ratio to sample mean {fmt(c1['ratio'], c1['ci'])}, to TabPFN {fmt(c2['ratio'], c2['ci'])}")
        say(f"  {lab} ({len(fl)} fields): " + "; ".join(parts))

    say("\n-- E9.7: TabPFN-GP at n = 5 on all blocks (not only where TabPFN-KpR predicted) --")
    allgp = cov[(cov.variant == "J1p") & (cov.n == 5) & cov.cov80.notna()]
    for k in ("field", "zone"):
        e, ci, n = S.coverage(allgp, "J1p", k); e2, ci2, _ = S.coverage(main_set[main_set.n == 5], "J1p", k)
        say(f"  {k}: all blocks {fmt(e, ci)} ({int((S.kind_of(allgp) == k).sum())} means); common set {fmt(e2, ci2)}")

    say("\n-- E9.8: share of TabPFN-GP's summed CRPS by n (common set; per task, as in the ratio) --")
    g = common[common.variant == "J1p"]
    for k in ("field", "zone"):
        q = g[S.kind_of(g) == k]; tot = q.groupby(["field", "target"]).crps.transform("sum"); share = (q.crps / tot).groupby(q.n).sum() / q.groupby(["field", "target"]).ngroups
        say(f"  {k}: " + "; ".join(f"n={n}: {v:.3f}" for n, v in share.items()))

    say("\n-- E13: comparators with a factor for each n, set by the rule on the factor-setting data (each external task once) --")
    NS = (5, 10, 15, 20, 30, 50, 75, 100)
    fn = {}
    for v in list(comp) + ["DB"]:
        fn[v] = {}
        for nn in NS:
            sub = first[first.n == nn]
            if sub[(sub.variant == v) & ~sub.failed].empty:
                continue
            fn[v][nn], _ = factor_rule(sub, v)
        say(f"  {names[v]:12s} factors: " + "; ".join(f"n={k}: {x:.2f}" for k, x in fn[v].items()))
    tag = lambda d: d.assign(variant=np.where(d.variant.isin(list(fn)), d.variant + "@" + d.n.astype(str), d.variant))   # noqa: E731
    rr = tag(raw[raw.n.isin(NS)])
    fac_n = {"J1p": S.KAPPA, **{f"{v}@{k}": x for v in fn for k, x in fn[v].items()}}
    crps_n = {"J1p": S.KAPPA, **{f"{v}@{k}": (x if crps_fac[v] != 1.0 else 1.0) for v in fn for k, x in fn[v].items()}}   # each keeps its CRPS version
    sc_c = scoring.score(rr, fac_n); sc_r = scoring.score(rr, crps_n); sc_c["crps"] = sc_r["crps"].to_numpy()
    sc_c["variant"] = sc_c.variant.str.split("@").str[0]
    five = ["J1p"] + list(comp)
    cm = scoring.common_blocks(sc_c, five); cm = cm[cm.variant.isin(five)]
    dbn = sc_c[(sc_c.variant == "DB") & sc_c.cov80.notna()]
    for v in list(comp) + ["DB"]:
        s = cm if v != "DB" else pd.concat([cm[cm.variant == "J1p"], dbn])
        parts = []
        for k in ("field", "zone"):
            est, b, nf = S.rise(s, v, k); c = S.crps_contrast(s, "J1p", v, k)
            byn = s[(s.variant == v) & (S.kind_of(s) == k) & s.cov80.notna()].groupby("n").cov80.mean()
            e_c, ci_c, n_c = S.coverage(s, v, k)
            csv_rows.append(dict(analysis="E13", what="coverage", model=names[v], variant=v, kind=k, est=e_c, lo=ci_c[0], hi=ci_c[1], n=n_c))
            parts.append(f"{k}: D {est:+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}], coverage by n "
                         + " ".join(f"{x:.2f}" for x in byn.to_numpy()) + f", CRPS ratio TabPFN-GP / it {fmt(c['ratio'], c['ci'])}")
            csv_rows.append(dict(analysis="E13", what="crps_ratio", model=names[v], variant=v, kind=k, est=c["ratio"], lo=c["ci"][0], hi=c["ci"][1], n=c["fields"]))
            csv_rows.append(dict(analysis="E13", what="D", model=names[v], variant=v, kind=k, est=est, lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), n=nf))
        say(f"  {names[v]}: " + "\n      ".join(parts))
    (OUT / "report5.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(OUT / "report5.csv", index=False)


if __name__ == "__main__":
    main()
