"""E14 of docs/exploratory/plan_E14.md (CPU): coverage of means over fewer and more locations of the same area, and the dense limit.

  rows:    SOILGP_DATA=<work>/data SOILGP_RUN=<work>/run python -m experiments.exploratory.dense_limit rows <out> [workers]
  report:  python -m experiments.exploratory.dense_limit report <out> <work>/rows  -> <out>/dense_report.txt

Rows: for every split, property and block (field, zones) of the 43 fields, nested random sub-blocks of the block's unrevealed locations
(m = 10, 30, 100, 200 where smaller than the block; seed crc32(field|design|rep|target|block)) and the block itself ('all'), with
TabPFN-GP (J1p, rebuilt as in further_models_rows.py), the shared-error-only model (O1), TabPFN (TAB) and the sample mean (DB), stored with factor 1;
for J1p also the independent part of V (Vind).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
import zlib                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from scipy.stats import norm, t as tdist                                      # noqa: E402

MODE, OUT = sys.argv[1], Path(sys.argv[2])
from experiments.study import run as S                                        # noqa: E402

if MODE == "rows":
    block_cases = S.setup(S.evaluated())
    from experiments.pipeline import block_rows                               # noqa: E402
    from experiments.core.block_model import PRIORS                           # noqa: E402
    from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent   # noqa: E402

LEVELS = (10, 30, 100, 200)
KEY = ["field", "design", "rep", "target", "block"]


def rows_of(case):
    F = S.F
    Sx, tr, te, y = case["S"], case["tr"], case["te"], case["y"]
    n, yq, mT, sT, m_oof = len(tr), y[te], case["mT"], case["sT"], case["m_oof"]
    blocks, revealed = block_cases.blocks_of(case); gate = n >= F.GATE_N
    key = dict(field=case["field"], design=case["design"], rep=case["rep"], target=case["target"], n=n)
    sub = {}
    for b, i in blocks.items():
        order = np.random.default_rng(zlib.crc32(f"{case['field']}|{case['design']}|{case['rep']}|{case['target']}|{b}".encode())).permutation(i)
        for m in LEVELS:
            if m < len(i):
                sub[f"{b}|{m}"] = np.sort(order[:m])
        sub[f"{b}|all"] = i
    rows = []

    def add(variant, name, M, V, df=np.inf, Vind=np.nan):
        b, level = name.split("|"); i = sub[name]
        rows.append(dict(key, block=b, level=level, mB=len(blocks[b]), m=len(i), nB=len(revealed[b]), variant=variant, M=float(M),
                         V=float(V), df=float(df), Vind=float(Vind), obs=float(yq[i].mean())))
    try:
        g = ResidualField(PRIORS["POP"], field_extent(Sx)).fit(Sx[tr], y[tr] - m_oof, block_range_floor(Sx[tr]))
        mom, vbar = g.block_moments(Sx[te], sub)
        ind = np.clip(sT ** 2 - vbar, 0, None)
        for name, i in sub.items():
            mb, vb = mom[name]; vi = float(np.sum(ind[i])) / len(i) ** 2
            add("J1p", name, float(mT[i].mean()) + (mb if gate else 0.0), vi + vb, Vind=vi)
    except Exception:
        pass
    r = y[tr] - m_oof; rbar, s2 = float(r.mean()), float(r.var(ddof=1))
    for name, i in sub.items():
        b = name.split("|")[0]; m = len(i); mTB = float(mT[i].mean()); vT = float(np.sum(sT[i] ** 2)) / m ** 2
        add("O1", name, mTB + (rbar if gate else 0.0), s2 / n + vT)
        add("TAB", name, mTB, vT)
        rr = revealed[b]
        if len(rr) >= 2:
            yr = y[rr]; add("DB", name, yr.mean(), np.var(yr, ddof=1) * (1 / len(rr) + 1 / m), df=len(rr) - 1)
    return rows


def field_job(field):
    t0 = time.monotonic(); rows = []
    for _, case in block_rows.cases("study", field):
        if case is not None and case["m_oof"] is not None:
            rows += rows_of(case)
    pd.DataFrame(rows).to_csv(OUT / f"dense_{field}.csv", index=False)
    return field, len(rows), time.monotonic() - t0


# ------------------------------------------------------------------------------------------------------------------------- report
def report(rows_dir):
    _, factors, _, names = S.methods()
    tab = [v for v, g in S.methods()[0].items() if g == "TabPFN"][0]
    fac = {"J1p": factors["J1p"], "O1": factors["O1"], "TAB": factors[tab], "DB": factors["DB"]}
    label = {"J1p": "TabPFN-GP", "O1": "Shared-error-only", "TAB": "TabPFN", "DB": "Sample mean"}
    d = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("dense_*.csv"))], ignore_index=True)
    d = d[np.isfinite(d.M) & np.isfinite(d.V) & (d.V > 1e-18)].copy()
    q = np.where(np.isfinite(d.df), tdist.ppf(0.9, d.df.clip(lower=1)), norm.ppf(0.9))
    d["phi"] = d.variant.map(fac); d["hw"] = q * d.phi * np.sqrt(d.V)
    d["cov"] = (np.abs(d.obs - d.M) <= d.hw).astype(float)
    d["kind"] = np.where(d.block == "field", "field", "zone")
    lines = []

    def say(s=""):
        lines.append(s); print(s, flush=True)

    say("=== E14 (docs/exploratory/plan_E14.md): means over fewer and more locations of the same area; 80 % intervals with the study's factors ===")
    # check against the study's rows
    st = pd.concat([pd.read_csv(p, low_memory=False, usecols=KEY + ["variant", "M", "V", "failed"])
                    for p in sorted(Path(rows_dir).glob("rows_study_*.csv"))], ignore_index=True)
    st = st[(st.variant == "J1p") & ~st.failed.astype(bool)]
    a = d[(d.variant == "J1p") & (d.level == "all")].merge(st, on=KEY, suffixes=("", "_s"))
    say(f"check: TabPFN-GP with all locations against the study's rows: {len(a)} of {int(((d.variant == 'J1p') & (d.level == 'all')).sum())}"
        f" matched; largest relative difference M {float((abs(a.M - a.M_s) / abs(a.M_s).clip(lower=1e-12)).max()):.1e}, "
        f"V {float((abs(a.V - a.V_s) / a.V_s).max()):.1e}")
    say(f"rows: {len(d)}; fields {d.field.nunique()}")
    piv = d.pivot_table(index=KEY + ["variant"], columns="level", values="cov").reset_index()
    mb = d[d.level == "all"].set_index(KEY + ["variant"]).mB
    piv = piv.join(mb, on=KEY + ["variant"])
    piv["kind"] = np.where(piv.block == "field", "field", "zone")

    def pooled(g, col):
        s = g.groupby("field")[col].agg(["sum", "count"]); idx = S.boot_idx(len(s))
        est, b = S.ratio_ci(s["sum"].to_numpy(), s["count"].to_numpy(), idx)
        return est, np.percentile(b, [2.5, 97.5]), len(s)

    def paired(g, lev):
        x = g.assign(dd=g["all"] - g[lev]); s = x.groupby("field").dd.agg(["sum", "count"]); idx = S.boot_idx(len(s))
        est, b = S.ratio_ci(s["sum"].to_numpy(), s["count"].to_numpy(), idx)
        return est, np.percentile(b, [2.5, 97.5]), len(s)

    for kind, levels in (("field", ("10", "30", "100", "200")), ("zone", ("10", "30"))):
        say(f"\n-- {kind} means: coverage with m locations and with all of them, on the same splits (and D = all minus m) --")
        for v in ("J1p", "O1", "TAB", "DB"):
            for lev in levels:
                g = piv[(piv.kind == kind) & (piv.variant == v)].dropna(subset=[lev, "all"])
                if g.empty:
                    continue
                c1, i1, k = pooled(g, lev); c2, i2, _ = pooled(g, "all"); dd, di, _ = paired(g, lev)
                say(f"  {label[v]:17s} m = {lev:>3s}: {c1:.3f} [{i1[0]:.3f}, {i1[1]:.3f}]  all: {c2:.3f} [{i2[0]:.3f}, {i2[1]:.3f}]  "
                    f"D {dd:+.3f} [{di[0]:+.3f}, {di[1]:+.3f}]  ({len(g)} areas, {k} fields; median m_B {int(g.mB.median())})")
        cls = ((0, 100), (101, 200), (201, 10 ** 6)) if kind == "field" else ((11, 29), (30, 99), (100, 10 ** 6))
        say(f"  by m_B class, TabPFN-GP, D = all minus m = 10:")
        for lo, hi in cls:
            g = piv[(piv.kind == kind) & (piv.variant == "J1p") & piv.mB.between(lo, hi)].dropna(subset=["10", "all"])
            if g.empty:
                continue
            dd, di, k = paired(g, "10"); c2, i2, _ = pooled(g, "all")
            say(f"    m_B {lo}-{hi if hi < 10 ** 6 else 'max'}: D {dd:+.3f} [{di[0]:+.3f}, {di[1]:+.3f}]; coverage with all {c2:.3f} "
                f"({len(g)} areas, {k} fields)")

    say("\n-- dense limit: TabPFN-GP with all locations; share of V from the independent part, and coverage with V_inf = V - that part --")
    j = d[(d.variant == "J1p") & (d.level == "all")].copy()
    j["share"] = j.Vind / j.V
    j["cov_inf"] = (np.abs(j.obs - j.M) <= norm.ppf(0.9) * j.phi * np.sqrt((j.V - j.Vind).clip(lower=1e-18))).astype(float)
    for kind, cls in (("field", ((0, 100), (101, 200), (201, 10 ** 6))), ("zone", ((3, 9), (10, 29), (30, 99), (100, 10 ** 6)))):
        for lo, hi in cls:
            g = j[(j.kind == kind) & j.mB.between(lo, hi)]
            if g.empty:
                continue
            c, ci, k = pooled(g, "cov"); ci_, cii, _ = pooled(g, "cov_inf")
            say(f"  {kind} m_B {lo}-{hi if hi < 10 ** 6 else 'max'}: independent share median {g.share.median():.3f} (90th pct "
                f"{g.share.quantile(0.9):.3f}); coverage {c:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], with V_inf {ci_:.3f} [{cii[0]:.3f}, {cii[1]:.3f}] "
                f"({len(g)} areas, {k} fields)")
    (OUT / "dense_report.txt").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if MODE == "rows":
        workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        todo = [f for f in S.evaluated() if not (OUT / f"dense_{f}.csv").exists()]
        with Pool(min(workers, len(todo))) as pool:
            for field, k, sec in pool.imap_unordered(field_job, todo):
                print(f"{field}: {k} rows, {sec:.0f} s", flush=True)
    else:
        report(sys.argv[3])


if __name__ == "__main__":
    main()
