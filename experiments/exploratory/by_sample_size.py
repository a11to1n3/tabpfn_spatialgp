"""Exploratory analyses E1 and E2 of docs/exploratory/plan_E1-E2.md, on the saved area rows of the study run (CPU only).

  python -m experiments.exploratory.by_sample_size <work>/rows <out>

E1: CRPS and 80 % width ratios TabPFN-GP / comparator by sample size (the study's factors and CRPS versions; area sets of the report).
E2: the same with each comparator's factor tuned on these results for each sample size and kind of area (CRPS factor: minimises the
comparator's CRPS aggregate; coverage factor: smallest grid value with pooled coverage >= 0.80). TabPFN-GP keeps kappa = 1.15.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import beta as Bfun
from scipy.stats import norm, t as tdist

ROWS, OUT = sys.argv[1], Path(sys.argv[2])
from experiments.study import run as S                                        # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.core.aggregate import boot_fields, proper_by_field          # noqa: E402

KEY = ["field", "design", "rep", "target", "block"]
GRID = np.round(np.arange(1, 401) * 0.05, 2)                                 # 0.05, 0.10, ..., 20.00
Z80 = 1.2815515655446004
NS = (5, 10, 15, 20, 30, 50, 75, 100)
lines = []


def say(s=""):
    lines.append(s); print(s, flush=True)


def fmt(e, ci):
    return f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def kmask(s, kind):
    return (s.block == "field") == (kind == "field")


def paired(s, v, kind, col):
    """Areas of one kind where TabPFN-GP and v both have `col`: TabPFN-GP's value (a) and v's M, V, df, obs, n."""
    q = s[kmask(s, kind)]
    a = q[q.variant == "J1p"].set_index(KEY)[col]
    b = q[q.variant == v].set_index(KEY)[["M", "V", "df", "obs", "n", col]]
    t = b.join(a.rename("a"), how="inner")
    t = t[t["a"].notna() & t[col].notna()]
    return t.reset_index()


def stat(t, col_b):
    """Ratio a / b with the study's aggregation (per task log ratio of means, mean over properties, geometric mean over fields)."""
    u = t[["field", "target", "a", col_b]].rename(columns={col_b: "b"})
    Lf = proper_by_field(u); est, ci = boot_fields(Lf)
    return est, ci, int((Lf < 0).sum()), len(Lf)


def crps_grid(t, phi):
    """CRPS of v's predictive with its standard deviation multiplied by each phi (rows x len(phi))."""
    sd = np.sqrt(np.clip(t.V.to_numpy(), 0, None))[:, None] * phi[None, :]
    y, mu, df = t.obs.to_numpy()[:, None], t.M.to_numpy()[:, None], t.df.to_numpy()
    z = (y - mu) / sd
    out = np.empty_like(z)
    g = ~np.isfinite(df)
    out[g] = (sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))[g]
    if (~g).any():
        nu = np.where(g, 3.0, df)[:, None] * np.ones_like(z)
        c = sd * (z * (2 * tdist.cdf(z, nu) - 1) + 2 * tdist.pdf(z, nu) * (nu + z ** 2) / (nu - 1)
                  - 2 * np.sqrt(nu) / (nu - 1) * Bfun(0.5, nu - 0.5) / Bfun(0.5, nu / 2) ** 2)
        out[~g] = c[~g]
    return out


def half_q(t):
    df = t.df.to_numpy(); g = ~np.isfinite(df)                               # normal predictives; else a t with df degrees of freedom
    return np.where(g, Z80, tdist.ppf(0.9, np.where(g, 2.0, df)))


def crps_objective(t, C):
    """Mean over fields of the mean over properties of log(mean CRPS of the task), for each column of C."""
    task = t.groupby(["field", "target"]).ngroup().to_numpy(); nt = task.max() + 1
    cnt = np.bincount(task, minlength=nt)
    tm = np.stack([np.bincount(task, weights=C[:, j], minlength=nt) for j in range(C.shape[1])], 1) / cnt[:, None]
    tf = t.groupby(["field", "target"]).field.first().to_numpy()
    fid = pd.factorize(tf)[0]; nf = fid.max() + 1
    fm = np.stack([np.bincount(fid, weights=np.log(tm[:, j]), minlength=nf) for j in range(C.shape[1])], 1) / np.bincount(fid)[:, None]
    return fm.mean(0)


def main():
    comp, factors, crps_f, names = S.methods()
    raw = S.load_rows(ROWS, S.CODES)
    cov, common, db, pair_o1 = S.area_sets(raw, comp, factors, crps_f)
    sets = {v: common for v in comp}
    sets["DB"] = pd.concat([common[common.variant == "J1p"], db], ignore_index=True)
    order = list(comp) + ["DB"]
    say("=== Exploratory analyses E1 and E2 (docs/exploratory/plan_E1-E2.md; specified after the study's results had been seen) ===")

    say("\n-- check: aggregated over all n, the ratios must equal report.txt --")
    for kind in ("field", "zone"):
        for v in order:
            c = scoring.contrast(sets[v], "J1p", v, kind); w = scoring.contrast(sets[v], "J1p", v, kind, col="width")
            say(f"  {kind:5s} {names[v]:12s} CRPS {fmt(c['ratio'], c['ci'])}  width {fmt(w['ratio'], w['ci'])}")

    rows_csv = []
    say("\n-- E1: ratio TabPFN-GP / comparator by n, the study's factors [95 % interval over fields; fields lower / fields] --")
    for kind in ("field", "zone"):
        for v in order:
            parts_c, parts_w = [], []
            for n in NS:
                s = sets[v][sets[v].n == n]
                c = scoring.contrast(s, "J1p", v, kind); w = scoring.contrast(s, "J1p", v, kind, col="width")
                if c is None:
                    continue
                parts_c.append(f"{n}: {fmt(c['ratio'], c['ci'])} {c['better']}/{c['fields']}")
                parts_w.append(f"{n}: {fmt(w['ratio'], w['ci'])}")
                rows_csv.append(dict(analysis="E1", kind=kind, method=names[v], n=n, factor=factors.get(v, 1.0), crps_ratio=c["ratio"],
                                     crps_lo=c["ci"][0], crps_hi=c["ci"][1], lower=c["better"], fields=c["fields"], width_ratio=w["ratio"],
                                     width_lo=w["ci"][0], width_hi=w["ci"][1]))
            say(f"  {kind:5s} {names[v]:12s} CRPS  " + "; ".join(parts_c))
            say(f"  {kind:5s} {names[v]:12s} width " + "; ".join(parts_w))

    say("\n-- E2: comparators with a factor tuned on these results for each n and kind (TabPFN-GP: kappa = 1.15) --")
    for kind in ("field", "zone"):
        for v in order:
            tc_all, tw_all = [], []
            for n in NS:
                tc = paired(sets[v][sets[v].n == n], v, kind, "crps")
                tw = paired(sets[v][sets[v].n == n], v, kind, "width")
                if tc.empty:
                    continue
                # the recomputed scores must equal the stored ones at the study's factors
                assert np.allclose(crps_grid(tc, np.array([crps_f.get(v, 1.0)]))[:, 0], tc["crps"].to_numpy(), rtol=1e-9, atol=1e-12)
                assert np.allclose(2 * half_q(tw) * np.sqrt(np.clip(tw.V.to_numpy(), 0, None)) * factors.get(v, 1.0),
                                   tw["width"].to_numpy(), rtol=1e-9, atol=1e-12)
                C = crps_grid(tc, GRID); obj = crps_objective(tc, C); j = int(np.argmin(obj)); phi_c = float(GRID[j])
                tc = tc.assign(b=C[:, j]); tc_all.append(tc)
                ec, cic, lc, nfc = stat(tc, "b")
                # coverage factor: smallest grid value with pooled coverage >= 0.80 on the width areas
                sd = np.sqrt(np.clip(tw.V.to_numpy(), 0, None)); hq = half_q(tw); err = np.abs(tw.obs.to_numpy() - tw.M.to_numpy())
                covg = (err[:, None] <= hq[:, None] * sd[:, None] * GRID[None, :]).mean(0)
                k = int(np.argmax(covg >= 0.80)) if (covg >= 0.80).any() else len(GRID) - 1; phi_w = float(GRID[k])
                tw = tw.assign(b=2 * hq * sd * phi_w); tw_all.append(tw)
                ew, ciw, lw, nfw = stat(tw, "b")
                gp_cov = cov[(cov.variant == "J1p") & kmask(cov, kind)].set_index(KEY).cov80.reindex(
                    tw.set_index(KEY).index).mean()
                edge = " (grid edge)" if j in (0, len(GRID) - 1) or k == len(GRID) - 1 else ""
                say(f"  {kind:5s} {names[v]:12s} n={n:3d}{edge}: CRPS factor {phi_c:5.2f} ratio {fmt(ec, cic)} lower {lc}/{nfc}; "
                    f"coverage factor {phi_w:5.2f} (coverage {covg[k]:.3f}; TabPFN-GP {gp_cov:.3f}) width ratio {fmt(ew, ciw)}")
                rows_csv.append(dict(analysis="E2", kind=kind, method=names[v], n=n, factor=phi_c, crps_ratio=ec, crps_lo=cic[0],
                                     crps_hi=cic[1], lower=lc, fields=nfc, cov_factor=phi_w, cov_at_factor=float(covg[k]),
                                     gp_coverage=float(gp_cov), width_ratio=ew, width_lo=ciw[0], width_hi=ciw[1]))
            ec, cic, lc, nfc = stat(pd.concat(tc_all), "b"); ew, ciw, lw, nfw = stat(pd.concat(tw_all), "b")
            say(f"  {kind:5s} {names[v]:12s} all n: CRPS ratio {fmt(ec, cic)} lower {lc}/{nfc}; width ratio {fmt(ew, ciw)} lower {lw}/{nfw}")
            rows_csv.append(dict(analysis="E2", kind=kind, method=names[v], n=0, crps_ratio=ec, crps_lo=cic[0], crps_hi=cic[1], lower=lc,
                                 fields=nfc, width_ratio=ew, width_lo=ciw[0], width_hi=ciw[1]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explore_report.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows_csv).to_csv(OUT / "explore_by_n.csv", index=False)


if __name__ == "__main__":
    main()
