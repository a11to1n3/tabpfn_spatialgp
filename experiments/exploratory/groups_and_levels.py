"""Exploratory analyses E3 and E4 of docs/exploratory/plan_E3-E5.md, on the saved area rows of the study run (CPU only).

  python -m experiments.exploratory.groups_and_levels <work>/rows <out>

E3: rise in coverage D by group of fields and by soil property, every method, field and zone means.
E4: coverage of central 50, 90 and 95 % intervals (80 % as a check), every method with its single factor, all n, n <= 10, n >= 30.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

ROWS, OUT = sys.argv[1], Path(sys.argv[2])
from experiments.study import run as S                                        # noqa: E402

lines, csv_rows = [], []


def say(s=""):
    lines.append(s); print(s, flush=True)


def fmt(e, lo, hi):
    return f"{e:.3f} [{lo:.3f}, {hi:.3f}]"


def kind_of(s):
    return np.where(s.block == "field", "field", "zone")


def pooled(s, col, kind):
    """Pooled share of `col` over areas of one kind, 95 % percentile interval over fields (as study.run.coverage)."""
    q = s[(kind_of(s) == kind) & s[col].notna()]
    if q.empty:
        return np.nan, np.nan, np.nan, 0
    g = q.groupby("field")[col].agg(["sum", "count"]); idx = S.boot_idx(len(g))
    a, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    b = a[idx].sum(1) / c[idx].sum(1)
    return float(a.sum() / c.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(g)


def main():
    comp, factors, crps_f, names = S.methods()
    raw = S.load_rows(ROWS, S.CODES)
    cov, common, db, pair_o1 = S.area_sets(raw, comp, factors, crps_f)
    main_set = pd.concat([common, db], ignore_index=True)
    allv = ["J1p", "O1"] + list(comp) + ["DB"]
    src = lambda v: pair_o1 if v == "O1" else main_set                        # noqa: E731
    say("=== Exploratory analyses E3 and E4 (docs/exploratory/plan_E3-E5.md; specified after the study's results had been seen) ===")

    # ---------------------------------------------------------------- E3
    say("\n-- check: pooled D equals report.txt --")
    for v in allv:
        for kind in ("field", "zone"):
            est, b, k = S.rise(src(v), v, kind)
            say(f"  {kind:5s} {names[v]:17s} D = {est:+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] ({k} fields)")
    for label, key, values in (("group", "field", ("Uniformity trials", "SensRes", "Midwest", "New fields", "Further fields")),
                               ("property", "target", ("SOC_target", "Clay_target", "pH_target"))):
        say(f"\n-- E3: rise in coverage D by {label} [95 % interval over the {label}'s fields; fields with both bands] --")
        for val in values:
            say(f"  {val}")
            for v in allv:
                parts = []
                for kind in ("field", "zone"):
                    s = src(v)
                    s = s[s.field.map(S.group) == val] if label == "group" else s[s.target == val]
                    est, b, k = S.rise(s, v, kind)
                    if not np.isfinite(est):
                        parts.append(f"{kind} -"); continue
                    lo, hi = np.percentile(b, [2.5, 97.5])
                    parts.append(f"{kind} {est:+.3f} [{lo:+.3f}, {hi:+.3f}] ({k})")
                    csv_rows.append(dict(analysis="E3", by=label, value=val, method=names[v], kind=kind, D=est, lo=lo, hi=hi, fields=k))
                say(f"    {names[v]:17s} " + "; ".join(parts))

    # ---------------------------------------------------------------- E4
    say("\n-- E4: coverage of central intervals with each method's single factor [95 % interval over fields] --")
    for v in allv:
        s = src(v); s = s[(s.variant == v) & s.cov80.notna()].copy()
        sd = np.sqrt(np.clip(s.V.to_numpy(), 0, None)) * factors.get(v, 1.0)
        err = np.abs(s.obs.to_numpy() - s.M.to_numpy()); df = s.df.to_numpy(); gauss = ~np.isfinite(df)
        for lev in (0.5, 0.8, 0.9, 0.95):
            p = (1 + lev) / 2
            q = np.where(gauss, norm.ppf(p), tdist.ppf(p, np.where(gauss, 2.0, df)))
            s[f"c{int(lev * 100)}"] = (err <= q * sd).astype(float)
        assert np.array_equal(s["c80"].to_numpy(), s["cov80"].to_numpy()), names[v]      # reproduces the report's 80 % coverage
        for kind in ("field", "zone"):
            parts = []
            for lev in (50, 80, 90, 95):
                row = []
                for band, sel in (("all", s.n > 0), ("n<=10", s.n.isin(S.SMALL)), ("n>=30", s.n.isin(S.LARGE))):
                    e, lo, hi, k = pooled(s[sel], f"c{lev}", kind)
                    row.append(f"{band} {fmt(e, lo, hi)}")
                    csv_rows.append(dict(analysis="E4", method=names[v], kind=kind, level=lev, band=band, coverage=e, lo=lo, hi=hi, fields=k))
                parts.append(f"{lev} %: " + "; ".join(row))
            say(f"  {kind:5s} {names[v]:17s}\n      " + "\n      ".join(parts))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explore2_report.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(OUT / "explore2.csv", index=False)


if __name__ == "__main__":
    main()
