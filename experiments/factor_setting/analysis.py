"""Analysis of the area rows of a set. reproduce.sh runs it on the first set (the factor-setting data), as a check of the code and of the rows;
compare.py checks its report against expected/factor_setting/report_first.txt.

  python -m experiments.factor_setting.analysis --rows DIR --set final|first --selection selection.json --calibration calibration.json --out DIR
  -> DIR/report_<set>.txt and DIR/results_<set>.json

Primary (LimeSoDa, common set = blocks where TabPFN-GP and the four selected model-based comparators are all scored):
  E1  TabPFN-GP's pooled 80 % coverage of field and of zone means; met if the 95 % bootstrap interval over fields lies within [0.75, 0.85].
  E2  CRPS ratio TabPFN-GP / comparator for TabPFN, TabPFN-KpR, block RK, WH (common set) and the sample mean (paired on its own blocks),
      field and zone means; one-sided exact Wilcoxon over fields, Holm over the 10 contrasts at 0.05; comparators in their most favourable
      version; non-inferiority (upper bound < 1.02) reported where superiority is not shown.
Secondary S1-S6; no pass/fail.
Reporting only: zone sizes and revealed samples per zone at each n,
failed cases per method and design, and the common set's losses at each n. These use no scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.core import config as F
from experiments.core import scoring
from experiments.factor_setting.set_factors import read_rows

MODEL_GROUPS = ("TabPFN", "TabPFN-KpR", "Block RK", "WH")


def fmt(c, word="better"):
    return "-" if c is None else f"{c['ratio']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}], fields {word} {c['better']}/{c['fields']}"


def analyse(rows_dir, set_name, selection, calibration, out):
    raw = read_rows(rows_dir, set_name); lines, res = [], {"set": set_name, "selection": selection}
    sel = {g: selection[g] for g in F.GROUPS}
    cal = calibration["comparators"]
    crps_factor = {cal[g]["variant"]: (cal[g]["factor"] if cal[g]["crps_version"] == "calibrated" else 1.0) for g in cal}
    cov_factor = {cal[g]["variant"]: cal[g]["factor"] for g in cal}
    ours_factor = {"J1pL": F.KAPPA_OURS}                                     # J1pL rows are unscaled; J1pK already includes kappa
    # tasks analysed: LimeSoDa all three properties; external primary tasks (Meuse's copies in the first set are dropped here)
    keep = (raw.src == "limesoda") | raw[["field", "target"]].apply(tuple, axis=1).isin(F.PRIMARY_EXTERNAL)
    keep_dense_ext = raw[["field", "target"]].apply(tuple, axis=1).isin(F.DENSE_EXTERNAL)
    data = raw[keep]; dense_ext = raw[keep_dense_ext]
    r_crps = scoring.score(data, {**crps_factor, **ours_factor}); r_cov = scoring.score(data, {**cov_factor, **ours_factor})
    r_unc = scoring.score(data, ours_factor)
    lim_c, lim_v = r_crps[r_crps.src == "limesoda"], r_cov[r_cov.src == "limesoda"]
    model_variants = ["J1pK"] + [sel[g] for g in MODEL_GROUPS]
    common_c, common_v = scoring.common_blocks(lim_c, model_variants), scoring.common_blocks(lim_v, model_variants)

    def say(s=""):
        lines.append(s); print(s, flush=True)

    say(f"=== Analysis of the area rows, set '{set_name}' ===")
    say(f"selected comparator variants: {sel}")
    say("factors: " + ", ".join(f"{g}: {cal[g]['variant']} x{cal[g]['factor']:.2f} (CRPS version {cal[g]['crps_version']})" for g in cal))
    # ---------------------------------------------------------------- failures and the common set
    say("\n-- failures (share of blocks, LimeSoDa) and common set --")
    for v in model_variants + ["DB"]:
        s = data[(data.src == "limesoda") & (data.variant == v)]
        say(f"  {v:14s} failed {s.failed.mean():.4f} of {len(s)} blocks")
    nb_all = int(scoring.kind_rows(lim_c, "field").query("variant == 'J1pK'").shape[0])
    nb_common = int(scoring.kind_rows(common_c, "field").query("variant == 'J1pK'").shape[0])
    say(f"  common set: {nb_common} of {nb_all} field blocks")
    res["common_field_blocks"], res["all_field_blocks"] = nb_common, nb_all
    # ---------------------------------------------------------------- design and failures (reporting only)
    # Uses no scores: zone sizes and revealed samples come from the TabPFN rows (one per block); failures are counted per split x
    # property (a case fails for all its blocks); the common set's losses per n use the same blocks as E1 and E2.
    say("\n-- design and failures (reporting only) --")
    rep_d = {"zones": {}, "failed_cases": {}, "common_set_by_n": {}}
    for src in ("limesoda", "external"):
        z = data[(data.src == src) & (data.variant == "TAB") & (data.block != "field")]
        say(f"  zones, {src}: per n, number of zone blocks; unrevealed locations m_B (min, quartiles, max); revealed samples n_B "
            "(median; share with n_B = 0, >= 2, >= 3)")
        for n_, g in z.groupby("n"):
            q = np.percentile(g.mB, [0, 25, 50, 75, 100]).round(1)
            say(f"    n = {n_:3d}: {len(g):5d} zones; m_B {q[0]:g} / {q[1]:g} / {q[2]:g} / {q[3]:g} / {q[4]:g}; n_B median "
                f"{np.median(g.nB):g}; n_B = 0 {np.mean(g.nB == 0):.3f}, >= 2 {np.mean(g.nB >= 2):.3f}, >= 3 {np.mean(g.nB >= 3):.3f}")
            rep_d["zones"][f"{src}_n{n_}"] = {"zones": int(len(g)), "mB_quantiles": q.tolist(), "nB_median": float(np.median(g.nB)),
                                              "nB0": float(np.mean(g.nB == 0)), "nB2": float(np.mean(g.nB >= 2)), "nB3": float(np.mean(g.nB >= 3))}
    for src in ("limesoda", "external"):
        f = data[(data.src == src) & (data.block == "field") & data.variant.isin(model_variants + ["KpR", "RK_ins_loc", "WH_loc"])]
        tab = f.groupby(["variant", "design"]).failed.agg(["sum", "count"])
        say(f"  failed split x property cases, {src} (failed / total) per design:")
        for v, g in tab.groupby(level="variant"):
            cells = ", ".join(f"{d} {int(r['sum'])}/{int(r['count'])}" for (_, d), r in g.iterrows())
            say(f"    {v:14s} {cells}")
            rep_d["failed_cases"][f"{src}_{v}"] = {d: [int(r["sum"]), int(r["count"])] for (_, d), r in g.iterrows()}
    say("  common set by n (LimeSoDa): field blocks kept / all; zone blocks kept / all")
    for n_ in sorted(lim_c.n.unique()):
        a_f = int(scoring.kind_rows(lim_c[lim_c.n == n_], "field").query("variant == 'J1pK'").shape[0])
        k_f = int(scoring.kind_rows(common_c[common_c.n == n_], "field").query("variant == 'J1pK'").shape[0])
        a_z = int(scoring.kind_rows(lim_c[lim_c.n == n_], "zone").query("variant == 'J1pK'").shape[0])
        k_z = int(scoring.kind_rows(common_c[common_c.n == n_], "zone").query("variant == 'J1pK'").shape[0])
        say(f"    n = {n_:3d}: field {k_f}/{a_f}, zone {k_z}/{a_z}")
        rep_d["common_set_by_n"][int(n_)] = {"field": [k_f, a_f], "zone": [k_z, a_z]}
    res["design_report"] = rep_d
    # ---------------------------------------------------------------- E1
    say("\n-- E1: TabPFN-GP coverage (pooled; 95 % bootstrap over fields), LimeSoDa common set --")
    res["E1"] = {}
    for kind in ("field", "zone"):
        est, ci = scoring.pooled_coverage(common_c, "J1pK", kind)
        met = F.E1_BAND[0] <= ci[0] and ci[1] <= F.E1_BAND[1]
        res["E1"][kind] = {"coverage": est, "ci": ci, "met": bool(met)}
        say(f"  {kind} means: {est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] -> {'MET' if met else 'NOT MET'} (band {F.E1_BAND})")
    # ---------------------------------------------------------------- E2
    say("\n-- E2: CRPS ratio TabPFN-GP / comparator (Wilcoxon over fields, Holm over 10) --")
    contrasts, pvals = [], []
    for g in list(MODEL_GROUPS) + ["Sample mean"]:
        v = sel[g]
        for kind in ("field", "zone"):
            base = common_c if g != "Sample mean" else lim_c
            c = scoring.contrast(base, "J1pK", v, kind)
            p = scoring.wilcoxon_less(c["Lf"]) if c else np.nan
            contrasts.append((g, v, kind, c)); pvals.append(p)
    adj = scoring.holm(np.nan_to_num(pvals, nan=1.0)); res["E2"] = []
    for (g, v, kind, c), p, pa in zip(contrasts, pvals, adj):
        sup = bool(pa < F.ALPHA and c is not None and c["ratio"] < 1)
        noninf = bool(c is not None and c["ci"][1] < F.NONINF_MARGIN)
        res["E2"].append({"comparator": g, "variant": v, "kind": kind, "ratio": None if c is None else c["ratio"],
                          "ci": None if c is None else c["ci"], "fields_better": None if c is None else c["better"],
                          "fields": None if c is None else c["fields"], "p": p, "p_holm": float(pa), "superior": sup,
                          "noninferior_2pct": noninf})
        say(f"  vs {g:12s} ({v:12s}) {kind:5s}: {fmt(c)}; p = {p:.2g}, Holm {pa:.2g} -> "
            f"{'SUPERIOR' if sup else ('non-inferior (2 %)' if noninf else 'not shown')}")
    # ---------------------------------------------------------------- secondary analyses
    say("\n-- S1: n <= 20 --")
    small_c = common_c[common_c.n <= 20]; small_l = lim_c[lim_c.n <= 20]; res["S1"] = {}
    for kind in ("field", "zone"):
        est, ci = scoring.pooled_coverage(small_c, "J1pK", kind)
        say(f"  TabPFN-GP {kind} coverage {est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        res["S1"][f"coverage_{kind}"] = (est, ci)
        for g in list(MODEL_GROUPS) + ["Sample mean"]:
            c = scoring.contrast(small_c if g != "Sample mean" else small_l, "J1pK", sel[g], kind)
            say(f"    vs {g:12s} {kind}: {fmt(c)}"); res["S1"][f"{g}_{kind}"] = None if c is None else (c["ratio"], c["ci"])
    say("\n-- S2: transfer --")
    held = list(F.C.HELD_BACK); res["S2"] = {}
    for kind in ("field", "zone"):
        est, ci = scoring.pooled_coverage(common_c, "J1pK", kind, fields=held)
        say(f"  held-back fields, TabPFN-GP {kind} coverage {est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        for g in MODEL_GROUPS:
            say(f"    vs {g:12s} {kind}: {fmt(scoring.contrast(common_c, 'J1pK', sel[g], kind, fields=held))}")
    ext_c = r_crps[r_crps.src == "external"]; ext_common = scoring.common_blocks(ext_c, model_variants)
    for kind in ("field", "zone"):
        cov = float(scoring.kind_rows(ext_common, kind).query("variant == 'J1pK'").cov80.mean())
        say(f"  Oxford/Meuse primary tasks, TabPFN-GP {kind} coverage {cov:.3f}")
        for g in MODEL_GROUPS:
            if ext_common.empty:
                say("    (no Oxford/Meuse rows)"); break
            e = scoring.external(ext_common, "J1pK", sel[g], kind)
            say(f"    vs {g:12s} {kind}: {e['ratio']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}], per task " +
                ", ".join(f"{x:.3f}" for x in e["tasks"]))
    say("\n-- S3: interval width ratio TabPFN-GP / comparator (aggregated like the CRPS), with coverage --")
    for g in MODEL_GROUPS:
        for kind in ("field", "zone"):
            c = scoring.contrast(common_v, "J1pK", sel[g], kind, col="width")
            say(f"  vs {g:12s} {kind}: width {fmt(c, 'narrower')}; coverage {g} {scoring.pooled_coverage(common_v, sel[g], kind, boot=False)[0]:.3f}")
    say("\n-- S4: comparator coverage, calibrated | uncalibrated (LimeSoDa common set) --")
    unc_common = scoring.common_blocks(r_unc[r_unc.src == "limesoda"], model_variants)
    for g in MODEL_GROUPS:
        v = sel[g]
        say(f"  {g:12s} {v:12s} field {scoring.pooled_coverage(common_v, v, 'field', boot=False)[0]:.3f} | "
            f"{scoring.pooled_coverage(unc_common, v, 'field', boot=False)[0]:.3f};  zone "
            f"{scoring.pooled_coverage(common_v, v, 'zone', boot=False)[0]:.3f} | {scoring.pooled_coverage(unc_common, v, 'zone', boot=False)[0]:.3f}")
    say("\n-- S5: published comparator versions (uncalibrated): TAB, KpR, block RK as in the KpR authors' code (RK_ins_loc), WH local --")
    pub = scoring.common_blocks(r_unc[r_unc.src == "limesoda"], ["J1pK", "TAB", "KpR", "RK_ins_loc", "WH_loc"])
    for v in ("TAB", "KpR", "RK_ins_loc", "WH_loc"):
        for kind in ("field", "zone"):
            say(f"  vs {v:12s} {kind}: {fmt(scoring.contrast(pub, 'J1pK', v, kind))}; coverage {v} "
                f"{scoring.pooled_coverage(pub, v, kind, boot=False)[0]:.3f}")
    say("\n-- S6 --")
    # dense covariates: the 14 fields without spectra, plus Meuse and Oxford without colour
    for kind in ("field", "zone"):
        est, ci = scoring.pooled_coverage(common_c, "J1pK", kind, fields=list(F.DENSE_FIELDS))
        say(f"  dense covariates, 14 fields without spectra: TabPFN-GP {kind} coverage {est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        for g in MODEL_GROUPS:
            say(f"    vs {g:12s} {kind}: {fmt(scoring.contrast(common_c, 'J1pK', sel[g], kind, fields=list(F.DENSE_FIELDS)))}")
    de = scoring.common_blocks(scoring.score(dense_ext, {**crps_factor, **ours_factor}), model_variants)
    for kind in ("field", "zone"):
        cov = float(scoring.kind_rows(de, kind).query("variant == 'J1pK'").cov80.mean())
        say(f"  dense covariates, Meuse + Oxford without colour: TabPFN-GP {kind} coverage {cov:.3f}")
    nodisp = [f for f in F.C.LIMESODA if f not in F.DUMMY_COORDS]
    for kind in ("field", "zone"):
        est, ci = scoring.pooled_coverage(common_c, "J1pK", kind, fields=nodisp)
        say(f"  without SL.125/UL.120: TabPFN-GP {kind} coverage {est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    for kind in ("field", "zone"):
        c = scoring.contrast(lim_c, "J1pL", "J1pK", kind, fields=list(F.C.DEV_FIELDS))
        say(f"  leave-one-field-out prior (development fields), J1pL x{F.KAPPA_OURS:.2f} vs J1pK {kind}: {fmt(c)}; coverage J1pL "
            f"{scoring.pooled_coverage(lim_c[lim_c.field.isin(F.C.DEV_FIELDS)], 'J1pL', kind, boot=False)[0]:.3f}")
    # substitutes: a failed method gets TabPFN with independent errors; KpR's own fallback counts as its prediction
    sub = data.copy(); tab = sub[sub.variant == "TAB"].set_index(["field", "design", "rep", "target", "block"])[["M", "V"]]
    idx = sub.set_index(["field", "design", "rep", "target", "block"]).index
    repl = sub.failed & ~sub.own_fallback & sub.variant.isin(model_variants)
    sub.loc[repl, ["M", "V"]] = tab.reindex(idx[repl.to_numpy()]).to_numpy(); sub.loc[repl | sub.own_fallback, "failed"] = False
    sub_c = scoring.score(sub, {**crps_factor, **ours_factor}); sub_l = sub_c[sub_c.src == "limesoda"]
    for kind in ("field", "zone"):
        est, ci = scoring.pooled_coverage(sub_l, "J1pK", kind)
        say(f"  all blocks with substitutes: TabPFN-GP {kind} coverage {est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        for g in MODEL_GROUPS:
            say(f"    vs {g:12s} {kind}: {fmt(scoring.contrast(sub_l, 'J1pK', sel[g], kind))}")
    say("  field-averaged coverage (each field equal), TabPFN-GP: field "
        f"{scoring.field_mean_coverage(common_c, 'J1pK', 'field'):.3f}, zone {scoring.field_mean_coverage(common_c, 'J1pK', 'zone'):.3f}")
    say("  by n, field means: TabPFN-GP coverage | CRPS ratio vs " + ", ".join(MODEL_GROUPS))
    for n in sorted(common_c.n.unique()):
        s = common_c[common_c.n == n]
        cov = scoring.pooled_coverage(s, "J1pK", "field", boot=False)[0]
        rs = [scoring.contrast(s, "J1pK", sel[g], "field") for g in MODEL_GROUPS]
        say(f"    n = {n:3d}: {cov:.3f} | " + ", ".join("-" if c is None else f"{c['ratio']:.3f}" for c in rs))
    Path(out, f"report_{set_name}.txt").write_text("\n".join(lines) + "\n")
    Path(out, f"results_{set_name}.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--set", required=True, choices=["first", "final"])
    ap.add_argument("--selection", required=True); ap.add_argument("--calibration", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    analyse(a.rows, a.set, json.loads(Path(a.selection).read_text()), json.loads(Path(a.calibration).read_text()), a.out)


if __name__ == "__main__":
    main()
