"""E15-E17 of docs/exploratory/plan_E15-E17.md (CPU): TabPFN-GP without its factor; zones without a sample; zones by number of revealed samples.

  python -m experiments.exploratory.factor_and_zones <work>/rows <work>/variants expected/exploratory/factors_once.json <out>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROWS, VAR, FAC, OUT = (Path(a) for a in sys.argv[1:5])
from experiments.study import run as S                                        # noqa: E402
from experiments.core import scoring                                          # noqa: E402

lines, csv_rows = [], []
FURTHER = ("Z", "GPx", "GPnp", "GPna", "GPnn", "DE", "DEz")                    # rows in work/variants; the others in work/rows


def say(s=""):
    lines.append(s); print(s, flush=True)


def fmt(e, ci):
    return f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def rec(analysis, what, model, variant, kind, est, lo, hi, n):
    csv_rows.append(dict(analysis=analysis, what=what, model=model, variant=variant, est=est, lo=lo, hi=hi, n=n, kind=kind))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    once = json.loads(FAC.read_text()); fac, crps_f = once["factor"], once["crps_factor"]
    raw = S.load_rows(ROWS, S.CODES)
    var = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(VAR.glob("variants_study_*.csv"))], ignore_index=True)
    var["failed"] = var.failed.astype(bool); var = var[var.field.isin(S.CODES)]
    comp, factors, crps_fac, names = S.methods()
    cov, common, db, pair_o1 = S.area_sets(raw, comp, factors, crps_fac)
    main_set = pd.concat([common, db], ignore_index=True)
    zone = lambda s: s[s.block != "field"]                                     # noqa: E731

    def further(v, cov_factor=None):
        """TabPFN-GP and model v on the areas where both are scored; coverage with the factors, CRPS by the version rule (as E11)."""
        src = var if v in FURTHER else raw
        rows = pd.concat([raw[raw.variant == "J1p"], src[src.variant == v]], ignore_index=True)
        sc = scoring.score(rows, {"J1p": S.KAPPA, v: cov_factor or fac[v]}); sr = scoring.score(rows, {"J1p": crps_f["J1p"], v: crps_f[v]})
        sc["crps"] = sr["crps"].to_numpy()
        both = scoring.common_blocks(sc, ["J1p", v]); return both[both.variant.isin(["J1p", v])]

    say("=== E15-E17 (docs/exploratory/plan_E15-E17.md; specified after the study's results had been seen; exploratory) ===")
    say("\n-- check --")
    for k in ("field", "zone"):
        say(f"  TabPFN-GP coverage, {k} means: {fmt(*S.coverage(common, 'J1p', k)[:2])}")
    say(f"  TabPFN-GP coverage, zones without a sample: {fmt(*S.coverage(main_set[main_set.nB == 0], 'J1p', 'zone')[:2])}")
    for v in ("Z", "GPnp"):
        b = further(v); say(f"  {v} coverage: {S.coverage(b, v, 'field')[0]:.3f} | {S.coverage(b, v, 'zone')[0]:.3f}")

    # ------------------------------------------------------------------------------------------------------------ E15
    say("\n-- E15: TabPFN-GP without its factor (kappa = 1), on the areas scored by all methods --")
    s1 = scoring.score(common[common.variant == "J1p"].drop(columns=["crps", "cov80", "width"]).assign(variant="J1p_1"), {"J1p_1": 1.0})
    c1 = pd.concat([common, s1], ignore_index=True)
    for k in ("field", "zone"):
        e, ci, n = S.coverage(c1, "J1p_1", k); d, b, nf = S.rise(c1, "J1p_1", k); cc = S.crps_contrast(c1, "J1p", "J1p_1", k)
        lo, hi = np.percentile(b, [2.5, 97.5])
        say(f"  {k}: coverage {fmt(e, ci)}; D {d:+.3f} [{lo:+.3f}, {hi:+.3f}] ({nf} fields); CRPS ratio TabPFN-GP (kappa 1.15) / "
            f"TabPFN-GP (kappa 1) {fmt(cc['ratio'], cc['ci'])} ({cc['better']}/{cc['fields']} fields lower)")
        rec("E15", "coverage", "TabPFN-GP without its factor", "J1p_1", k, e, *ci, n)
        rec("E15", "D", "TabPFN-GP without its factor", "J1p_1", k, d, lo, hi, nf)
        rec("E15", "crps_ratio", "TabPFN-GP without its factor", "J1p_1", k, cc["ratio"], *cc["ci"], cc["fields"])
    by_n = pd.concat([common, s1])
    by_n = by_n[by_n.variant.isin(["J1p", "J1p_1"]) & by_n.cov80.notna()].assign(kind=lambda d: S.kind_of(d))
    say("  coverage by n (pooled): kappa 1.15 | kappa 1")
    for k in ("field", "zone"):
        t = by_n[by_n.kind == k].groupby(["n", "variant"]).cov80.mean().unstack("variant")
        say(f"    {k}: " + "; ".join(f"{n}: {r['J1p']:.3f} | {r['J1p_1']:.3f}" for n, r in t.iterrows()))
    m1 = pd.concat([main_set, s1], ignore_index=True); e, ci, n = S.coverage(m1[m1.nB == 0], "J1p_1", "zone")
    say(f"  zones without a sample: coverage {fmt(e, ci)}"); rec("E15", "coverage_nB0", "TabPFN-GP without its factor", "J1p_1", "zone", e, *ci, n)

    # ------------------------------------------------------------------------------------------------------------ E16
    say("\n-- E16: zones without a revealed sample (n_B = 0), each model with its own factor (as E11) --")
    models = [("Z", "Zone-effect model"), ("GPx", "Prior refitted without SL.125, UL.120"), ("GPnp", "No cross-field prior"),
              ("GPna", "No averaging"), ("GPnn", "Neither"), ("RK_oof_fix_v", "Block RK of held-out errors"),
              ("WH_loc", "WH with each field's own correlogram"), ("O1", "Shared-error-only, own factor")]
    for v, name in models:
        q = further(v); q = q[(q.block != "field") & (q.nB == 0)]
        e, ci, n = S.coverage(q, v, "zone"); eg, cig, _ = S.coverage(q, "J1p", "zone"); cc = S.crps_contrast(q, "J1p", v, "zone")
        say(f"  {name:40s} factor {fac[v]:.2f}: coverage {fmt(e, ci)} (TabPFN-GP on the same zones {fmt(eg, cig)}); CRPS ratio "
            f"TabPFN-GP / model {fmt(cc['ratio'], cc['ci'])} ({cc['better']}/{cc['fields']} fields lower); {int((q.variant == v).sum())} zone means")
        rec("E16", "coverage_nB0", name, v, "zone", e, *ci, n); rec("E16", "coverage_nB0_gp", name, v, "zone", eg, *cig, n)
        rec("E16", "crps_ratio_nB0", name, v, "zone", cc["ratio"], *cc["ci"], cc["fields"])
    q = zone(pair_o1)[zone(pair_o1).nB == 0]; e, ci, n = S.coverage(q, "O1", "zone")
    say(f"  {'Shared-error-only, kappa 1.15':40s}: coverage {fmt(e, ci)}"); rec("E16", "coverage_nB0", "Shared-error-only, kappa 1.15", "O1", "zone", e, *ci, n)

    # ------------------------------------------------------------------------------------------------------------ E17
    say("\n-- E17: zones by number of revealed samples n_B --")
    z = zone(cov); z = z[(z.variant == "J1p") & z.cov80.notna()]
    say("  zone means scored for TabPFN-GP: " + "; ".join(f"n_B = {k}: {int((z.nB == k).sum())}" for k in range(0, 6)) +
        f"; n_B >= 6: {int((z.nB >= 6).sum())}")
    zfur = further("Z")
    dez = scoring.score(var[var.variant == "DEz"], {"DEz": fac["DEz"]}); dez = zone(dez[dez.cov80.notna()])
    for lab, sel in (("1", lambda s: s.nB == 1), ("2", lambda s: s.nB == 2), (">=3", lambda s: s.nB >= 3)):
        q = zone(common)[sel(zone(common))]
        say(f"  n_B {lab}: {int((q.variant == 'J1p').sum())} zone means scored by all methods")
        for v in ["J1p"] + list(comp):
            e, ci, n = S.coverage(q, v, "zone"); txt = f"    {names[v]:17s} coverage {fmt(e, ci)}"
            rec("E17", "coverage", names[v], v, f"nB {lab}", e, *ci, n)
            if v != "J1p":
                cc = S.crps_contrast(q, "J1p", v, "zone"); rec("E17", "crps_ratio", names[v], v, f"nB {lab}", cc["ratio"], *cc["ci"], cc["fields"])
                txt += f"; CRPS ratio TabPFN-GP / it {fmt(cc['ratio'], cc['ci'])} ({cc['better']}/{cc['fields']} fields lower)"
            say(txt)
        qo = zone(pair_o1)[sel(zone(pair_o1))]; e, ci, n = S.coverage(qo, "O1", "zone")
        say(f"    {'Shared-error-only':17s} coverage {fmt(e, ci)} (kappa 1.15)"); rec("E17", "coverage", "Shared-error-only", "O1", f"nB {lab}", e, *ci, n)
        qz = zone(zfur)[sel(zone(zfur))]; e, ci, n = S.coverage(qz, "Z", "zone"); cc = S.crps_contrast(qz, "J1p", "Z", "zone")
        say(f"    {'Zone-effect model':17s} coverage {fmt(e, ci)} (factor {fac['Z']:.2f}); CRPS ratio TabPFN-GP / it {fmt(cc['ratio'], cc['ci'])} (as E11)")
        rec("E17", "coverage", "Zone-effect model", "Z", f"nB {lab}", e, *ci, n); rec("E17", "crps_ratio", "Zone-effect model", "Z", f"nB {lab}", cc["ratio"], *cc["ci"], cc["fields"])
        for name, v, s in (("Sample mean", "DB", zone(db)), ("Difference estimator, zones", "DEz", dez)):
            qs = s[sel(s)]
            if len(qs):
                e, ci, n = S.coverage(qs, v, "zone")
                say(f"    {name:17s} coverage {fmt(e, ci)} ({len(qs)} zone means)"); rec("E17", "coverage", name, v, f"nB {lab}", e, *ci, n)

    (OUT / "report7.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(OUT / "report7.csv", index=False)


if __name__ == "__main__":
    main()
