"""Exploratory analyses, scores: other values of kappa and factors chosen without the evaluated data, the band test for every comparator,
the offset-only model, the difference estimators, the plug-in analogue and cross-fitted block RK, coverage by property, Oxford and
Meuse intervals, the fitted spatial share and a design subgroup. Stored rows only; no refits.

  python -m experiments.sensitivity.models_report --stored ROWS --calibration CALIBRATION.json --raw RAW --out OUT
--stored: the area rows of the final set (pipeline/block_rows.py); --calibration: the calibration.json of the factor setting;
--raw: the rows of pipeline/further_models.py in RAW/first/ and RAW/final/, and the first set's area rows in RAW/first_rows/.

Scoring, aggregation and the factor rule are those of the study's code (core/scoring.py, set_factors.factor_for).
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.core import config as F
from experiments.core import scoring

warnings.filterwarnings("ignore")
KEY = ["field", "design", "rep", "target", "block"]
LINES, OUT = [], {}


def say(s=""):
    LINES.append(s); print(s, flush=True)


def read(files):
    return pd.concat([pd.read_csv(f, dtype={"err": str}) for f in files], ignore_index=True)


def factor_rule(r, variant, grid):
    """set_factors.factor_for with an explicit grid: smallest factor whose mean of pooled field and zone 80 % coverage reaches 0.80."""
    s = r[(r.variant == variant) & ~r.failed]
    sd = np.sqrt(np.clip(s.V.to_numpy(), 0, None)); df = s.df.to_numpy(); err = np.abs(s.obs.to_numpy() - s.M.to_numpy())
    q = np.where(np.isfinite(df), scoring.tdist.ppf(0.9, np.where(np.isfinite(df), df, 2.0)), F.Z80); field = (s.block == "field").to_numpy()
    for f in grid:
        c = err <= q * f * sd
        if (c[field].mean() + c[~field].mean()) / 2 >= F.COVERAGE_TARGET:
            return float(f)
    return float(grid[-1])


def scaled(rows, variant, factor_of_field, name):
    """Rows of `variant` renamed `name`, scored with a factor per field (dict field -> factor)."""
    s = rows[rows.variant == variant].copy(); s["variant"] = name
    f = s.field.map(factor_of_field).to_numpy(float)
    return scoring.score(s.assign(V=s.V * f ** 2))


def cov_ci(r, v, kind, fields=None):
    est, ci = scoring.pooled_coverage(r, v, kind, fields=fields)
    return est, ci, f"{est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def ratio_txt(c):
    return "-" if c is None else f"{c['ratio']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] ({c['better']}/{c['fields']})"


def ext_cov(r, v, kind, seed=0, B=10000):
    """Pooled coverage over the primary external tasks, with a percentile interval from resampling replicates within each task."""
    s = scoring.kind_rows(r, kind); s = s[(s.variant == v) & s.cov80.notna()]
    est = float(s.cov80.mean()); rng = np.random.default_rng(seed); per = []
    for (field, target), q in s.groupby(["field", "target"]):
        g = q.groupby("rep").cov80.agg(["sum", "count"]); per.append((g["sum"].to_numpy(), g["count"].to_numpy()))
    tot = np.zeros(B); cnt = np.zeros(B)
    for su, co in per:
        idx = rng.integers(0, len(su), (B, len(su))); tot += su[idx].sum(1); cnt += co[idx].sum(1)
    lo, hi = np.percentile(tot / cnt, [2.5, 97.5])
    return est, (float(lo), float(hi)), f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    for k in ("stored", "calibration", "raw", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    C = F.C; raw = Path(a.raw)
    KGRID = np.round(np.arange(1.00, 1.5001, 0.05), 2)                        # the grid for kappa
    cal = json.loads(Path(a.calibration).read_text())["comparators"]
    cov_factor = {cal[g]["variant"]: cal[g]["factor"] for g in cal}
    crps_factor = {cal[g]["variant"]: (cal[g]["factor"] if cal[g]["crps_version"] == "calibrated" else 1.0) for g in cal}
    SEL = ["TAB", "KpR4", "RK_ins_fix_v", "WH_fix"]

    # ------------------------------------------------------------------------------------------------ rows and checks
    fin = read(sorted(Path(a.stored).glob("rows_final_*.csv"))); first = read(sorted((raw / "first_rows").glob("rows_first_*.csv")))
    nfin = read(sorted((raw / "final").glob("models_final_*.csv"))); nfirst = read(sorted((raw / "first").glob("models_first_*.csv")))
    diag = read(sorted((raw / "final").glob("diagnostics_final_*.csv")))
    say("=== Exploratory analyses on the final splits (further_models.py rows and the stored rows) ===")
    say("\n-- checks: recomputed TAB and J1p against the stored rows (blocks matched, failures, largest relative difference of M, V) --")
    for name, new, ref in (("final", nfin, fin), ("first", nfirst, first)):
        for v in ("TAB", "J1p"):
            x = new[new.variant == v].set_index(KEY); y = ref[ref.variant == v].set_index(KEY); both = x.index.intersection(y.index)
            ok = both[~x.loc[both, "failed"].to_numpy() & ~y.loc[both, "failed"].to_numpy()]
            rel = max(float((np.abs(x.loc[ok, c] - y.loc[ok, c]) / np.abs(y.loc[ok, c]).clip(lower=1e-12)).max()) for c in ("M", "V"))
            say(f"  {name:5s} {v:4s}: {len(both)} of {len(x)} blocks matched ({len(y)} stored); failed {int(x.failed.sum())} vs "
                f"{int(y.failed.sum())}; max rel diff {rel:.1e}")
    NEW = ["O1", "DE", "DEz"]
    fin = pd.concat([fin, nfin[nfin.variant.isin(NEW)]], ignore_index=True)
    first = pd.concat([first, nfirst[nfirst.variant.isin(NEW)]], ignore_index=True)
    keep = (fin.src == "limesoda") | fin[["field", "target"]].apply(tuple, axis=1).isin(F.PRIMARY_EXTERNAL)
    data = fin[keep]
    r_cov = scoring.score(data, {**cov_factor}); r_crps = scoring.score(data, {**crps_factor})
    lim_cov, lim_crps = r_cov[r_cov.src == "limesoda"], r_crps[r_crps.src == "limesoda"]
    common_crps = scoring.common_blocks(lim_crps, ["J1pK"] + SEL); ckey = common_crps.set_index(KEY).index.unique()
    incommon = lambda r: r[r.set_index(KEY).index.isin(ckey)]                                     # noqa: E731
    common_cov = incommon(lim_cov)
    e1 = {k: cov_ci(common_crps, "J1pK", k)[2] for k in ("field", "zone")}
    say(f"  E1 reproduced on the common set ({int((common_crps.variant.eq('J1pK') & common_crps.block.eq('field')).sum())} field blocks): "
        f"field {e1['field']}, zone {e1['zone']}")
    k_check = factor_rule(first, "J1p", KGRID)
    say(f"  factor rule on the first set: J1p -> {k_check:.2f}; " + ", ".join(
        f"{cal[g]['variant']} -> {factor_rule(first, cal[g]['variant'], F.FACTOR_GRID):.2f} (chosen {cal[g]['factor']:.2f})" for g in cal))

    # ------------------------------------------------------------------------------------------------ 1. kappa
    say("\n-- 1. TabPFN-GP at other values of kappa (LimeSoDa common set; band [0.75, 0.85]; CRPS ratio to kappa = 1.10) --")
    base = incommon(data[(data.src == "limesoda") & data.variant.isin(["J1p", "J1pK"])]); OUT["kappa"] = {}
    for k in (1.00, 1.05, 1.10, 1.15, 1.20):
        s = base[base.variant == "J1p"].assign(variant="K"); sc = scoring.score(pd.concat([s, base[base.variant == "J1pK"]]), {"K": k})
        row = {}
        for kind in ("field", "zone"):
            est, ci, txt = cov_ci(sc, "K", kind); c = scoring.contrast(sc, "K", "J1pK", kind)
            row[kind] = {"coverage": est, "ci": ci, "in_band": bool(F.E1_BAND[0] <= ci[0] and ci[1] <= F.E1_BAND[1]), "crps_ratio": c["ratio"]}
            row[kind + "_txt"] = (f"{txt} {'in band' if row[kind]['in_band'] else 'OUT'} (bounds {ci[0]:.4f}, {ci[1]:.4f}); "
                                  f"CRPS {c['ratio']:.4f}")
        OUT["kappa"][f"{k:.2f}"] = row
        say(f"  kappa {k:.2f}: field {row['field_txt']} | zone {row['zone_txt']}")

    # ------------------------------------------------------------------------------------------------ 2. held-out kappa and factors
    say("\n-- 2. Factors chosen without the evaluated data (the factor rule on the first-set rows) --")
    dedup = ~((first.field == "OXFNC") | ((first.field == "MEUSE") & (first.target != "SOC_target")))
    say(f"  all first-set rows: kappa {factor_rule(first, 'J1p', KGRID):.2f}; each external task once: {factor_rule(first[dedup], 'J1p', KGRID):.2f}; "
        f"20 fields alone: {factor_rule(first[first.src == 'limesoda'], 'J1p', KGRID):.2f}")
    OUT["heldout"] = {}
    for label, rows_first in (("stored rows", first), ("each external task once", first[dedup])):
        kf = {f: factor_rule(rows_first[rows_first.field != f], "J1p", KGRID) for f in C.LIMESODA}
        sc = scaled(base, "J1p", kf, "LOFO"); sc = pd.concat([sc, scoring.score(base[base.variant == "J1pK"])])
        parts = []
        for kind in ("field", "zone"):
            est, ci, txt = cov_ci(sc, "LOFO", kind); c = scoring.contrast(sc, "LOFO", "J1pK", kind)
            parts.append(f"{kind} {txt} (CRPS vs kappa 1.10 {c['ratio']:.4f})")
            OUT["heldout"][f"lofo_{label}_{kind}"] = {"coverage": est, "ci": ci}
        vals = pd.Series(kf); OUT["heldout"][f"lofo_{label}_kappas"] = vals.value_counts().sort_index().to_dict()
        say(f"  kappa leaving out each field ({label}): values {dict(vals.value_counts().sort_index())}; " + "; ".join(parts))
    # external datasets: kappa without the dataset itself (stored rows)
    ext = r_cov[r_cov.src == "external"]; ext_common = scoring.common_blocks(scoring.score(data[data.src == "external"], crps_factor), ["J1pK"] + SEL)
    xkey = ext_common.set_index(KEY).index.unique()
    xbase = data[(data.src == "external") & data.variant.isin(["J1p"])]; xbase = xbase[xbase.set_index(KEY).index.isin(xkey)]
    kx = {"OXF": factor_rule(first[~first.field.isin(["OXF", "OXFNC"])], "J1p", KGRID),
          "MEUSE": factor_rule(first[first.field != "MEUSE"], "J1p", KGRID)}
    sx = scaled(xbase, "J1p", kx, "LODO")
    say(f"  kappa leaving out Oxford {kx['OXF']:.2f}, Meuse {kx['MEUSE']:.2f}; Oxford/Meuse primary tasks coverage: " +
        "; ".join(f"{k} {ext_cov(sx, 'LODO', k)[2]}" for k in ("field", "zone")))
    # development fields only
    kd = factor_rule(first[first.field.isin(C.DEV_FIELDS)], "J1p", KGRID)
    sd_ = scaled(base, "J1p", {f: kd for f in C.LIMESODA}, "DEV"); sdx = scaled(xbase, "J1p", {f: kd for f in ("OXF", "MEUSE")}, "DEV")
    say(f"  kappa from the 14 development fields only: {kd:.2f}; held-back fields: " +
        "; ".join(f"{k} {cov_ci(sd_, 'DEV', k, fields=list(C.HELD_BACK))[2]}" for k in ("field", "zone")) +
        "; Oxford/Meuse: " + "; ".join(f"{k} {ext_cov(sdx, 'DEV', k)[2]}" for k in ("field", "zone")))
    OUT["heldout"]["kappa_dev_only"] = kd; OUT["heldout"]["kappa_lodo"] = kx
    # comparators' factors, leaving out each field
    say("  comparators, factor leaving out each field (range of the 20 values) -> pooled coverage, against the chosen factor:")
    for g in cal:
        v = cal[g]["variant"]; ff = {f: factor_rule(first[first.field != f], v, F.FACTOR_GRID) for f in C.LIMESODA}
        src = incommon(data[data.variant == v]) if v != "DB" else data[(data.src == "limesoda") & (data.variant == "DB")]
        sc = scaled(src, v, ff, "LOFO"); reg = scaled(src, v, {f: cal[g]["factor"] for f in C.LIMESODA}, "REG")
        both = pd.concat([sc, reg])
        say(f"    {g:12s} {v:12s} factors {min(ff.values()):.2f}-{max(ff.values()):.2f}: " + "; ".join(
            f"{k} {cov_ci(both, 'LOFO', k)[2]} (chosen {cov_ci(both, 'REG', k)[2]})" for k in ("field", "zone")))

    # ------------------------------------------------------------------------------------------------ 3. band test for every comparator
    say("\n-- 3. Band test applied to every comparator with its chosen factor (LimeSoDa common set; sample mean on its own blocks) --")
    OUT["band"] = {}
    for g in cal:
        v = cal[g]["variant"]; src = common_cov if v != "DB" else lim_cov
        parts = []
        for kind in ("field", "zone"):
            est, ci, txt = cov_ci(src, v, kind); inb = F.E1_BAND[0] <= ci[0] and ci[1] <= F.E1_BAND[1]
            parts.append(f"{kind} {txt} {'in band' if inb else 'OUT'}"); OUT["band"][f"{v}_{kind}"] = {"coverage": est, "ci": ci, "in_band": bool(inb)}
        say(f"  {g:12s} {v:12s} x{cal[g]['factor']:.2f}: " + "; ".join(parts))

    # ------------------------------------------------------------------------------------------------ 4. further estimators
    say("\n-- 4. Further estimators: factor by the factor rule; coverage at factor 1 and at the own factor; CRPS ratio TabPFN-GP / "
        "estimator (most favourable version, as for the comparators); pairs with TabPFN-GP inside the LimeSoDa common set --")
    X = ["O1", "DE", "DEz", "PI_fix", "RK_oof_fix", "RK_oof_fix_v", "PI_loc", "RK_oof_loc", "RK_oof_loc_v"]
    lim_first = first[first.src == "limesoda"]; OUT["further"] = {}
    for v in X:
        fac = factor_rule(first, v, F.FACTOR_GRID)
        s = lim_first[lim_first.variant == v]; both = pd.concat([s, s.assign(variant=v + "__cal")], ignore_index=True)
        scd = scoring.score(both, {v + "__cal": fac})
        rcal = float(np.exp(np.mean([np.log(scoring.contrast(scd, v + "__cal", v, k)["ratio"]) for k in ("field", "zone")])))
        crps_f = fac if rcal < 1 else 1.0
        rows = incommon(data[(data.src == "limesoda") & data.variant.isin([v, "J1pK"])])
        sc1 = scoring.score(rows); scf = scoring.score(rows, {v: fac}); scc = scoring.score(rows, {v: crps_f})
        res = {"factor": fac, "crps_version": "own factor" if rcal < 1 else "factor 1"}
        say(f"  {v:12s} factor {fac:.2f} (CRPS version: {res['crps_version']}; on the first set calibrated/uncalibrated {rcal:.3f})")
        for kind in ("field", "zone"):
            e1_, c1, t1 = cov_ci(sc1, v, kind); ef, cf, tf = cov_ci(scf, v, kind); c = scoring.contrast(scc, "J1pK", v, kind)
            bn = scoring.kind_rows(scf, kind); bn = bn[bn.variant == v]
            byn = bn.groupby("n").cov80.mean()
            b1 = scoring.kind_rows(sc1, kind); b1 = b1[b1.variant == v].groupby("n").cov80.mean()
            res[kind + "_by_n_factor1"] = b1.round(3).to_dict()
            say(f"    {kind} by n at factor 1: " + " ".join(f"{int(n)}:{x:.3f}" for n, x in b1.items()) +
                f"; largest distance from 0.80 over n: factor 1 {np.abs(b1 - 0.8).max():.3f}, own factor {np.abs(byn - 0.8).max():.3f}")
            empty = scoring.kind_rows(scf, kind); empty = empty[(empty.variant == v) & (empty.nB == 0)].cov80.mean()
            res[kind] = {"cov_1": e1_, "cov_own": ef, "cov_own_ci": cf, "crps_ratio": None if c is None else c["ratio"],
                         "crps_ci": None if c is None else c["ci"], "by_n": byn.round(3).to_dict(), "empty_zones": float(empty)}
            say(f"    {kind}: coverage x1 {t1}; own {tf}; by n " + " ".join(f"{int(n)}:{x:.3f}" for n, x in byn.items()) +
                (f"; zones without samples {empty:.3f}" if kind == "zone" and np.isfinite(empty) else "") + f"; CRPS GP/{v} {ratio_txt(c)}")
            if kind == "field":
                rn = []
                for n in sorted(scc.n.unique()):
                    cc = scoring.contrast(scc[scc.n == n], "J1pK", v, "field"); rn.append(f"{int(n)}:{cc['ratio']:.3f}" if cc else f"{int(n)}:-")
                say("      CRPS GP/" + v + " by n (field): " + " ".join(rn)); res["crps_by_n_field"] = rn
        OUT["further"][v] = res
    # O1 at kappa = 1.10, the same factor as TabPFN-GP
    rows = incommon(data[(data.src == "limesoda") & data.variant.isin(["O1", "J1pK"])]); sc = scoring.score(rows, {"O1": F.KAPPA_OURS})
    say("  O1 at kappa = 1.10: " + "; ".join(f"{k} {cov_ci(sc, 'O1', k)[2]}, CRPS GP/O1 {ratio_txt(scoring.contrast(sc, 'J1pK', 'O1', k))}"
                                              for k in ("field", "zone")))
    for kind in ("field", "zone"):
        bn = scoring.kind_rows(sc, kind); bn = bn[bn.variant == "O1"]
        say(f"    {kind} by n: " + " ".join(f"{int(n)}:{x:.3f}" for n, x in bn.groupby("n").cov80.mean().items()) +
            (f"; zones without samples {bn[bn.nB == 0].cov80.mean():.3f}" if kind == "zone" else ""))
        rn = []
        for n in sorted(sc.n.unique()):
            cc = scoring.contrast(sc[sc.n == n], "J1pK", "O1", kind); rn.append(f"{int(n)}:{cc['ratio']:.3f}")
        say(f"    CRPS GP/O1 by n ({kind}): " + " ".join(rn))
    # TabPFN-GP's own coverage in zones without samples, for reference
    for kind in ("field", "zone"):
        g = scoring.kind_rows(common_crps, kind); g = g[g.variant == "J1pK"].groupby("n").cov80.mean()
        say(f"  TabPFN-GP {kind} by n: " + " ".join(f"{int(n)}:{x:.3f}" for n, x in g.items()) + f"; largest distance from 0.80 {np.abs(g - 0.8).max():.3f}")
    j = scoring.kind_rows(common_crps, "zone"); j = j[(j.variant == "J1pK") & (j.nB == 0)]
    say(f"  TabPFN-GP in zones without samples: {j.cov80.mean():.3f} ({len(j)} zone blocks)")
    # predictive standard deviation of TabPFN-GP relative to the offset-only model (both at kappa = 1)
    v2 = data[(data.src == "limesoda") & data.variant.isin(["J1p", "O1"])].pivot_table(index=KEY + ["n"], columns="variant", values="V")
    v2 = v2.dropna().reset_index(); v2["sd_ratio"] = np.sqrt(v2.J1p / v2.O1); v2["kind"] = np.where(v2.block == "field", "field", "zone")
    say("  sd(TabPFN-GP) / sd(offset only), median [quartiles] by n:")
    for kind in ("field", "zone"):
        g = v2[v2.kind == kind].groupby("n").sd_ratio.quantile([0.25, 0.5, 0.75]).unstack()
        say(f"    {kind}: " + " ".join(f"{int(n)}:{r[0.5]:.2f} [{r[0.25]:.2f}, {r[0.75]:.2f}]" for n, r in g.iterrows()))

    # ------------------------------------------------------------------------------------------------ 5. by soil property
    say("\n-- 5. Coverage by soil property (LimeSoDa; TabPFN-GP on the common set, the sample mean on its own blocks) --")
    db1 = scoring.score(data[(data.src == "limesoda") & (data.variant == "DB")])
    for t in C.TARGETS:
        parts = []
        for kind in ("field", "zone"):
            s = common_crps[common_crps.target == t]; d = lim_cov[lim_cov.target == t]; d1 = db1[db1.target == t]
            parts.append(f"{kind} GP {cov_ci(s, 'J1pK', kind)[2]}, sample mean x1.05 {cov_ci(d, 'DB', kind)[2]}, x1 {cov_ci(d1, 'DB', kind)[2]}")
        say(f"  {t:11s} " + " | ".join(parts))

    # ------------------------------------------------------------------------------------------------ 6. Oxford and Meuse
    say("\n-- 6. Oxford and Meuse, primary tasks: TabPFN-GP coverage (interval from resampling replicates within tasks) --")
    xs = scoring.score(data[data.src == "external"]); xs = xs[xs.set_index(KEY).index.isin(xkey)]
    say("  " + "; ".join(f"{k} {ext_cov(xs, 'J1pK', k)[2]}" for k in ("field", "zone")))

    # ------------------------------------------------------------------------------------------------ 7. fitted spatial share
    say("\n-- 7. Chosen model's fit, LimeSoDa final splits: spatial share (posterior mode; grid-weighted) and range / L, median [quartiles] by n --")
    d = diag[diag.src == "limesoda"]
    for n, g in d.groupby("n"):
        q = lambda c: f"{g[c].median():.2f} [{g[c].quantile(.25):.2f}, {g[c].quantile(.75):.2f}]"   # noqa: E731
        say(f"  n = {int(n):3d}: share mode {q('share_map')}, weighted {q('share_w')}; range/L {q('range_rel_map')}; "
            f"share mode < 0.2 in {np.mean(g.share_map < 0.2):.2f}, > 0.8 in {np.mean(g.share_map > 0.8):.2f}")
    # ------------------------------------------------------------------------------------------------ 8. offset only, ablation layout
    say("\n-- 8. Offset only at kappa = 1.10 in the layout of the ablation table (all LimeSoDa blocks; CRPS ratio variant / TabPFN-GP) --")
    allb = scoring.score(data[(data.src == "limesoda") & data.variant.isin(["O1", "J1pK"])], {"O1": F.KAPPA_OURS})
    for lab, sel_n in (("n <= 20", allb.n <= 20), ("n >= 30", allb.n >= 30)):
        s = allb[sel_n]; parts = []
        for v in ("J1pK", "O1"):
            cov = [scoring.pooled_coverage(s, v, k, boot=False)[0] for k in ("field", "zone")]
            rat = [scoring.contrast(s, v, "J1pK", k)["ratio"] if v != "J1pK" else 1.0 for k in ("field", "zone")]
            parts.append(f"{v}: coverage {cov[0]:.3f} {cov[1]:.3f}, CRPS ratio {rat[0]:.3f} {rat[1]:.3f}")
        nf = int(((s.variant == 'J1pK') & (s.block == 'field')).sum()); nz = int(((s.variant == 'J1pK') & (s.block != 'field')).sum())
        say(f"  {lab} ({nf} field, {nz} zone blocks): " + " | ".join(parts))
    z0 = scoring.kind_rows(allb, "zone"); z0 = z0[z0.nB == 0]
    say(f"  zones without samples, all blocks: TabPFN-GP {z0[z0.variant == 'J1pK'].cov80.mean():.3f}, O1 {z0[z0.variant == 'O1'].cov80.mean():.3f} "
        f"({int((z0.variant == 'J1pK').sum())} zone blocks)")
    # ------------------------------------------------------------------------------------------------ 9. design subgroup
    # LimeSoDa's documentation: SL.125 and UL.120 grid plus targeted sampling, H.138 adapted Latin hypercube plus uncertainty-guided
    # sampling, SM.40 stratified, NSW.52 random within k-means clusters plus stratified random, SC.93 conditioned Latin hypercube;
    # SP.231 systematic and simple random sampling, so it belongs to the subgroup.
    say("\n-- 9. Fields sampled on grids or at random: without the six datasets with targeted, stratified or conditioned Latin hypercube "
        "designs in LimeSoDa --")
    purposive = ["SL.125", "UL.120", "H.138", "SM.40", "NSW.52", "SC.93"]
    grid = [f for f in C.LIMESODA if f not in purposive]
    say(f"  {len(grid)} fields: {', '.join(grid)}")
    for kind in ("field", "zone"):
        parts = [f"TabPFN-GP coverage {cov_ci(common_crps, 'J1pK', kind, fields=grid)[2]}"]
        for v in SEL + ["DB"]:
            base_ = common_crps if v != "DB" else lim_crps
            c_all = scoring.contrast(base_, "J1pK", v, kind); c_g = scoring.contrast(base_, "J1pK", v, kind, fields=grid)
            parts.append(f"vs {v} {c_g['ratio']:.3f} (all 20: {c_all['ratio']:.3f})")
        say(f"  {kind}: " + "; ".join(parts))
    (out / "models_report.txt").write_text("\n".join(LINES) + "\n")
    (out / "models_results.json").write_text(json.dumps(OUT, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


if __name__ == "__main__":
    main()
