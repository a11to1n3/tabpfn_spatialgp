"""Exploratory analysis E6 of docs/exploratory/plan_E6.md: TabPFN-GP with a Student-t shape (nu = n - 1, 80 % interval unchanged) on the 43 fields.

  python -m experiments.exploratory.student_t <work>/rows <out>

The t version ('GP-t') uses TabPFN-GP's rows (stored with kappa = 1) on the report's area set: location M, scale
s = kappa * z_0.9 * V^(1/2) / t_nu(0.9), nu = n - 1, so its central 80 % interval is TabPFN-GP's.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

ROWS, OUT = sys.argv[1], Path(sys.argv[2])
from experiments.study import run as S                                        # noqa: E402
from experiments.core import scoring                                          # noqa: E402

LEV = (50, 80, 90, 95)
lines, csv_rows = [], []


def say(s=""):
    lines.append(s); print(s, flush=True)


def fmt(e, ci):
    return f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def pooled(s, col, kind):
    q = s[(S.kind_of(s) == kind) & s[col].notna()]
    g = q.groupby("field")[col].agg(["sum", "count"]); idx = S.boot_idx(len(g))
    a, c = g["sum"].to_numpy(float), g["count"].to_numpy(float); b = a[idx].sum(1) / c[idx].sum(1)
    return float(a.sum() / c.sum()), (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))), len(g)


def main():
    comp, factors, crps_f, names = S.methods()
    raw = S.load_rows(ROWS, S.CODES)
    cov, common, db, pair_o1 = S.area_sets(raw, comp, factors, crps_f)
    gp = common[common.variant == "J1p"].copy()
    ok = gp.crps.notna().to_numpy()
    n = gp.n.to_numpy(float); nu = n - 1; sdv = np.sqrt(np.clip(gp.V.to_numpy(), 0, None)); err = np.abs(gp.obs - gp.M).to_numpy()
    s_t = S.KAPPA * norm.ppf(0.9) * sdv / tdist.ppf(0.9, nu)
    t = gp.copy(); t["variant"] = "GPt"
    for lev in LEV:
        p = (1 + lev / 100) / 2
        gp[f"c{lev}"] = np.where(ok, (err <= S.KAPPA * norm.ppf(p) * sdv).astype(float), np.nan)
        t[f"c{lev}"] = np.where(ok, (err <= tdist.ppf(p, nu) * s_t).astype(float), np.nan)
        gp[f"w{lev}"] = np.where(ok, 2 * S.KAPPA * norm.ppf(p) * sdv, np.nan)
        t[f"w{lev}"] = np.where(ok, 2 * tdist.ppf(p, nu) * s_t, np.nan)
    t["crps"] = np.where(ok, scoring.crps_t(gp.obs.to_numpy(), gp.M.to_numpy(), s_t, nu), np.nan)
    assert np.array_equal(gp.c80[ok].to_numpy(), gp.cov80[ok].to_numpy()), "TabPFN-GP's 80 % coverage not reproduced"
    assert np.array_equal(t.c80[ok].to_numpy(), gp.c80[ok].to_numpy()), "the t version changed an 80 % interval"
    say("=== Exploratory analysis E6 (docs/exploratory/plan_E6.md; specified after the study's results had been seen) ===")
    say(f"TabPFN-GP rows on the report's area set: {len(gp)} ({int(ok.sum())} scored); 80 % intervals of the t version identical: yes")

    say("\n-- coverage [95 % interval over fields] --")
    for kind in ("field", "zone"):
        for name, s in (("TabPFN-GP", gp), ("TabPFN-GP, t shape", t)):
            parts = []
            for lev in LEV:
                row = []
                for band, sel in (("n<=10", s.n.isin(S.SMALL)), ("n>=30", s.n.isin(S.LARGE)), ("all", s.n > 0)):
                    e, ci, k = pooled(s[sel], f"c{lev}", kind)
                    row.append(f"{band} {fmt(e, ci)}")
                    csv_rows.append(dict(analysis="E6", what="coverage", method=name, kind=kind, level=lev, band=band, est=e, lo=ci[0],
                                         hi=ci[1], fields=k))
                parts.append(f"{lev} %: " + "; ".join(row))
            say(f"  {kind:5s} {name}\n      " + "\n      ".join(parts))

    say("\n-- rise in coverage D (n >= 30 minus n <= 10, fields with both) --")
    for kind in ("field", "zone"):
        for name, s, v in (("TabPFN-GP", gp, "J1p"), ("TabPFN-GP, t shape", t, "GPt")):
            parts = []
            for lev in (50, 95):
                est, b, k = S.rise(s.assign(cov80=s[f"c{lev}"]), v, kind); lo, hi = np.percentile(b, [2.5, 97.5])
                parts.append(f"{lev} %: {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                csv_rows.append(dict(analysis="E6", what="rise", method=name, kind=kind, level=lev, band="", est=est, lo=lo, hi=hi, fields=k))
            say(f"  {kind:5s} {name:20s} " + "; ".join(parts))

    say("\n-- CRPS ratio, t version / method [95 % interval over fields] --")
    both = pd.concat([common, t], ignore_index=True)
    for kind in ("field", "zone"):
        parts = []
        for v in ["J1p"] + list(comp) + ["DB"]:
            s = both if v != "DB" else pd.concat([t, db], ignore_index=True)
            c = S.crps_contrast(s, "GPt", v, kind)
            parts.append(f"{names[v]} {fmt(c['ratio'], c['ci'])}")
            csv_rows.append(dict(analysis="E6", what="crps_ratio", method=names[v], kind=kind, level=np.nan, band="", est=c["ratio"],
                                 lo=c["ci"][0], hi=c["ci"][1], fields=c["fields"]))
        say(f"  {kind:5s} " + "; ".join(parts))

    say("\n-- width of 95 % intervals, t version / TabPFN-GP [95 % interval over fields] --")
    w = pd.concat([gp.assign(width=gp.w95), t.assign(width=t.w95)], ignore_index=True)
    for kind in ("field", "zone"):
        for band, sel in (("n<=10", w.n.isin(S.SMALL)), ("n>=30", w.n.isin(S.LARGE)), ("all", w.n > 0)):
            c = S.crps_contrast(w[sel], "GPt", "J1p", kind, col="width")
            say(f"  {kind:5s} {band:6s} {fmt(c['ratio'], c['ci'])}")
            csv_rows.append(dict(analysis="E6", what="width95_ratio", method="TabPFN-GP", kind=kind, level=95, band=band, est=c["ratio"],
                                 lo=c["ci"][0], hi=c["ci"][1], fields=c["fields"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explore3_report.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(OUT / "explore3.csv", index=False)


if __name__ == "__main__":
    main()
