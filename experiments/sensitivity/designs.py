"""Exploratory: TabPFN-GP under other sampling designs, and its sensitivity to the random seeds.

designs: on the 20 fields, n = 8 and n = 20 revealed locations drawn by
  srs      simple random sampling (as in the study; at n = 20 the final splits themselves, replicates 120-129)
  strat    stratified random sampling: n/4 locations at random in each of the four zones
  clust    n/4 locations around each of four random seed locations (nearest-neighbour growth, as the cluster designs of core/datasets.py)
  sys      a spatially spread sample: the location nearest the centre of each of n k-means clusters of the coordinates (random start)
  tran     the n locations nearest a straight line through the centroid of the locations, at a random angle
10 replicates each (seeded by crc32('field|design|rep'), replicates 120-129), three properties. Each split: TabPFN with cross-fitted
residuals, as point_model.our_model computes them; then TabPFN-GP (kappa = 1.10), the offset-only model (kappa = 1.10), the difference
estimator with the field's residuals (field means) and with the zone's own residuals (zone means; t intervals without a factor), the
sample mean (t intervals without a factor), and for the stratified design the stratified estimator of the field mean (zone weights from the
unrevealed locations; t intervals with n - 4 degrees of freedom, without a factor).

seeds: on four fields at n = 5 and 20 (the final splits, replicates 120-129), TabPFN-GP with the seed of TabPFN and of the fold split
set to 1, 2, 3 and 4 besides seed 0: the spread of the interval endpoints over the five seeds, relative to the half-width.

  python -m experiments.sensitivity.designs points --out DIR [--fields ...]     (GPU)  -> DIR/points/ours/<design>/<field>/<target>/rep<r>.npz
  python -m experiments.sensitivity.designs rows --out DIR [--workers 14]       (CPU)  -> DIR/rows_designs_<field>.csv
  python -m experiments.sensitivity.designs seeds --out DIR [--fields ...]      (GPU)  -> DIR/seeds_<field>.csv
  python -m experiments.sensitivity.designs report --out DIR --final-rows DIR          -> DIR/report_designs.txt
--final-rows: the stored rows of the final set (read-only), for the simple random samples with n = 20 and the seeds' replicate 0.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one thread, as in the study's runs
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import time                                                                   # noqa: E402
import zlib                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from sklearn.cluster import KMeans                                            # noqa: E402
from sklearn.model_selection import KFold                                     # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.sensitivity import models_report as RR                     # noqa: E402
from experiments.core.common import bars, covariates, interval_sd, load_field, point_file, save_npz, tabpfn  # noqa: E402
from experiments.sensitivity.models_report import KEY                         # noqa: E402
from experiments.factor_setting.set_factors import read_rows                  # noqa: E402
from experiments.core.block_model import PRIORS, block_model                  # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402

C = F.C
KINDS = ("srs", "strat", "clust", "sys", "tran")
SIZES = (8, 20)
REPS = range(120, 130)
SEED_FIELDS = ("BB.250", "G.150", "MG.112", "NSW.52")
lines = []


def say(s=""):
    lines.append(s); print(s, flush=True)


def zone_labels(S):
    return KMeans(S_.ZONE_K, random_state=0, n_init=10).fit_predict(S)       # the clustering of zone_blocks


def design_rows(data, field, kind, n, rep):
    """Revealed rows of one split of a design (srs, strat, clust, sys, tran) with n locations."""
    S = np.asarray(data["coordinates"], float); N = len(S)
    if N - n < F.MIN_TEST:
        raise ValueError(f"{kind}{n} infeasible for {field}")
    rng = np.random.default_rng(zlib.crc32(f"{field}|{kind}{n}|{rep}".encode()))
    if kind == "srs":
        return np.sort(rng.choice(N, n, replace=False))
    if kind == "strat":
        z = zone_labels(S); out = []
        for j in range(S_.ZONE_K):
            idx = np.flatnonzero(z == j); out += list(rng.choice(idx, min(n // S_.ZONE_K, len(idx)), replace=False))
        return np.sort(np.array(out))
    if kind == "clust":
        seeds = rng.choice(N, 4, replace=False); chosen = set(seeds.tolist())
        order = [np.argsort(np.linalg.norm(S - S[s], axis=1)) for s in seeds]; ptr = [0] * 4; j = 0
        while len(chosen) < n:
            o = order[j % 4]
            while o[ptr[j % 4]] in chosen:
                ptr[j % 4] += 1
            chosen.add(int(o[ptr[j % 4]])); j += 1
        return np.array(sorted(chosen))
    if kind == "sys":
        centres = KMeans(n, random_state=int(rng.integers(2 ** 31)), n_init=1).fit(S).cluster_centers_; chosen = []
        for c in centres:
            for k in np.argsort(np.linalg.norm(S - c, axis=1)):
                if k not in chosen:
                    chosen.append(int(k)); break
        return np.sort(np.array(chosen))
    if kind == "tran":
        a = rng.uniform(0, np.pi); d = np.array([np.cos(a), np.sin(a)]); X = S - S.mean(0)
        return np.sort(np.argsort(np.abs(X[:, 0] * d[1] - X[:, 1] * d[0]), kind="stable")[:n])
    raise ValueError(kind)


def tabpfn_outputs(data, tr, te, target, seed=S_.RANDOM_STATE):
    """TabPFN's predictive distribution at te and its cross-fitted means at tr (point_model.our_model's TabPFN part; seed of TabPFN and
    of the fold split)."""
    make = lambda: tabpfn(n_estimators=4, target_transform=False, seed=seed)  # noqa: E731
    xa, xb = covariates(data, tr, te, with_coords=True); yt = data["Y"][:, C.TARGETS.index(target)][tr]
    bq = bars(make().fit(xa, yt), xb); mean_T, _ = bq.moments(); qs = bq.quantiles(np.array(C.QUANTILES))
    m_oof, v_oof = np.empty(len(tr)), np.empty(len(tr))
    for a, b in KFold(min(S_.OOF_FOLDS, len(tr)), shuffle=True, random_state=seed).split(xa):
        if np.ptp(yt[a]) == 0:
            m_oof[b], v_oof[b] = yt[a][0], max(float(np.var(yt)), 1e-12); continue
        m_oof[b], v_oof[b] = bars(make().fit(xa[a], yt[a]), xa[b]).moments()
    return dict(none_mean=mean_T, none_quantiles=qs, bb_m_oof=m_oof, bb_v_oof=v_oof)


def splits(fields):
    for field in fields:
        data = load_field(field)
        for kind in KINDS:
            for n in SIZES:
                if kind == "srs" and n == 20:                                 # the final splits (stored rows)
                    continue
                for rep in REPS:
                    try:
                        yield field, data, f"{kind}{n}", rep, design_rows(data, field, kind, n, rep)
                    except ValueError:
                        continue


def stage_points(out, fields):
    root = Path(out) / "points"
    for field, data, design, rep, tr in splits(fields or list(C.LIMESODA)):
        te = np.setdiff1d(np.arange(len(data["coordinates"])), tr)
        for target in C.TARGETS:
            path = point_file(root, "ours", design, field, target, rep)
            if path.exists() or not np.all(np.isfinite(data["Y"][:, C.TARGETS.index(target)])):
                continue
            t0 = time.monotonic(); save_npz(path, train_rows=tr, **tabpfn_outputs(data, tr, te, target))
            print(f"{design} {field} {target} rep={rep} {time.monotonic() - t0:.1f} s", flush=True)


def block_rows(key, S, tr, te, y, ours, kinds=("J1p", "O1", "DE", "DEz", "DB")):
    """Rows of TabPFN-GP (J1p, kappa = 1), the offset-only model, the difference estimators and the sample mean (unscaled V)."""
    blocks, revealed = block_cases.blocks_of(dict(S=S, tr=tr, te=te)); n = len(tr); yq = y[te]; gate = n >= F.GATE_N
    mT, sT = ours["none_mean"], interval_sd(ours["none_quantiles"]); r = y[tr] - ours["bb_m_oof"]; pos = {g: k for k, g in enumerate(tr)}
    rows = []

    def add(variant, b, M, V, df=np.inf):
        i = blocks[b]; failed = not (np.isfinite(M) and np.isfinite(V) and V > 1e-18)
        rows.append(dict(key, n=n, m=len(te), block=b, mB=len(i), nB=len(revealed[b]), variant=variant, M=float(M), V=float(V),
                         df=float(df), obs=float(yq[i].mean()), failed=failed))

    if "J1p" in kinds:
        try:
            for b, (M, V, _) in block_model(PRIORS["POP"], S, tr, te, y, ours, blocks, sT, gate).items():
                add("J1p", b, M, V)
        except Exception:
            for b in blocks:
                add("J1p", b, np.nan, np.nan)
    rbar, s2 = float(r.mean()), float(r.var(ddof=1))
    for b, i in blocks.items():
        m = len(i); mTB = float(mT[i].mean()); vind = float(np.sum(sT[i] ** 2)) / m ** 2
        if "O1" in kinds:
            add("O1", b, mTB + (rbar if gate else 0.0), s2 / n + vind)
        if "DE" in kinds and b == "field":
            add("DE", b, mTB + rbar, s2 * (1 / n + 1 / m), df=n - 1)
        rb = r[[pos[j] for j in revealed[b]]]
        if "DEz" in kinds and b != "field" and len(rb) >= 2:
            add("DEz", b, mTB + float(rb.mean()), float(rb.var(ddof=1)) * (1 / len(rb) + 1 / m), df=len(rb) - 1)
        yr = y[revealed[b]]
        if "DB" in kinds and len(yr) >= 2:
            add("DB", b, float(yr.mean()), float(yr.var(ddof=1)) * (1 / len(yr) + 1 / m), df=len(yr) - 1)
    return rows


def stratified_row(key, S, tr, te, y):
    """The stratified estimator of the field's unrevealed mean under stratified random sampling by zone: sum_h (m_h / m) ybar_{R_h},
    error variance sum_h (m_h / m)^2 S_h^2 (1 / n_h + 1 / m_h), t with n - H degrees of freedom (every zone needs two revealed samples)."""
    z = zone_labels(S); m = len(te); M = V = 0.0
    for h in range(S_.ZONE_K):
        r, u = tr[z[tr] == h], te[z[te] == h]
        if len(u) == 0:
            continue
        if len(r) < 2:
            return []
        w = len(u) / m; M += w * float(y[r].mean()); V += w ** 2 * float(y[r].var(ddof=1)) * (1 / len(r) + 1 / len(u))
    return [dict(key, n=len(tr), m=m, block="field", mB=m, nB=len(tr), variant="DBs", M=M, V=V, df=float(len(tr) - S_.ZONE_K),
                 obs=float(y[te].mean()), failed=not (np.isfinite(M) and V > 1e-18))]


def field_rows(job):
    field, out = job; root = Path(out) / "points"; rows = []
    for f, data, design, rep, tr in splits([field]):
        S = np.asarray(data["coordinates"], float); te = np.setdiff1d(np.arange(len(S)), tr)
        for target in C.TARGETS:
            p = point_file(root, "ours", design, f, target, rep)
            if not p.exists():
                continue
            ours = dict(np.load(p)); assert np.array_equal(ours["train_rows"], tr)
            key = dict(field=f, design=design, rep=rep, target=target); y = data["Y"][:, C.TARGETS.index(target)]
            rows += block_rows(key, S, tr, te, y, ours)
            if design.startswith("strat"):
                rows += stratified_row(key, S, tr, te, y)
    pd.DataFrame(rows).to_csv(Path(out) / f"rows_designs_{field}.csv", index=False)
    return field, len(rows)


def stage_rows(out, workers):
    with Pool(min(workers, len(C.LIMESODA))) as pool:
        for field, n in pool.imap_unordered(field_rows, [(f, out) for f in C.LIMESODA]):
            print(f"{field}: {n} rows", flush=True)


def stage_seeds(out, fields):
    for field in fields or SEED_FIELDS:
        rows = []
        for f, data, design, rep, tr in block_cases.iter_splits("final", [field]):
            if design not in ("n5", "n20"):
                continue
            S = np.asarray(data["coordinates"], float); te = np.setdiff1d(np.arange(len(S)), tr)
            for target in C.TARGETS:
                y = data["Y"][:, C.TARGETS.index(target)]
                if not np.all(np.isfinite(y)):
                    continue
                for seed in (1, 2, 3, 4):
                    t0 = time.monotonic(); ours = tabpfn_outputs(data, tr, te, target, seed=seed)
                    rows += block_rows(dict(field=f, design=design, rep=rep, target=target, seed=seed), S, tr, te, y, ours, kinds=("J1p",))
                    print(f"seed {seed} {design} {f} {target} rep={rep} {time.monotonic() - t0:.1f} s", flush=True)
        pd.DataFrame(rows).to_csv(Path(out) / f"seeds_{field}.csv", index=False)


def report(out, final_rows):
    say("=== Exploratory: other sampling designs, and the seeds of TabPFN and of the fold split ===")
    d = pd.concat([pd.read_csv(f) for f in sorted(Path(out).glob("rows_designs_*.csv"))], ignore_index=True)
    reg = read_rows(final_rows, "final"); reg = reg[(reg.src == "limesoda")]
    srs20 = reg[(reg.design == "n20") & reg.variant.isin(["J1p"])].assign(design="srs20")
    rev = d.assign(failed=d.failed.astype(bool))
    # simple random samples with n = 20: the final splits; O1, DE, DEz and DB recomputed from the stored point outputs
    extra = []
    for field in C.LIMESODA:
        for f, data, design, rep, tr in block_cases.iter_splits("final", [field]):
            if design != "n20":
                continue
            S = np.asarray(data["coordinates"], float); te = np.setdiff1d(np.arange(len(S)), tr)
            for target in C.TARGETS:
                y = data["Y"][:, C.TARGETS.index(target)]
                case = block_cases.load_case("final", f, data, design, rep, tr, target) if np.all(np.isfinite(y)) else None
                if case is not None:
                    extra += block_rows(dict(field=f, design="srs20", rep=rep, target=target), S, tr, te, y, case["ours"],
                                        kinds=("O1", "DE", "DEz", "DB"))
    allr = pd.concat([rev, srs20[rev.columns.intersection(srs20.columns)], pd.DataFrame(extra)], ignore_index=True)
    allr = allr.assign(failed=allr.failed.astype(bool), src="limesoda")
    allr = allr.assign(variant=allr.variant.replace({"J1p": "J1pK"}))
    sc = scoring.score(allr, {"J1pK": F.KAPPA_OURS, "O1": F.KAPPA_OURS})
    say("coverage of 80 % intervals (field means / zone means / zones without a revealed sample), 20 fields, 10 replicates, 3 properties;"
        " TabPFN-GP and the offset-only model at kappa = 1.10; difference estimators and sample mean as t intervals without a factor")
    for design in [f"{k}{n}" for n in SIZES for k in KINDS]:
        s = sc[sc.design == design]
        if s.empty:
            continue
        nos = s[(s.block == "field") | (s.nB == 0)]
        cells = []
        for v, fv in (("J1pK", "J1pK"), ("O1", "O1"), ("DEz", "DE"), ("DB", "DB")):
            fc = scoring.pooled_coverage(s, fv, "field", boot=False)[0]; zc = scoring.pooled_coverage(s, v, "zone", boot=False)[0]
            nc = scoring.pooled_coverage(nos, v, "zone", boot=False)[0] if v in ("J1pK", "O1") else float("nan")
            cells.append(f"{v if v != 'DEz' else 'DE/DEz'} {fc:.3f}/{zc:.3f}/{nc:.3f}")
        c1 = scoring.contrast(s, "J1pK", "O1", "zone"); c2 = scoring.contrast(s, "J1pK", "DB", "field")
        say(f"  {design:8s} " + "; ".join(cells) + f" | CRPS TabPFN-GP/offset-only zones {RR.ratio_txt(c1)}; /sample mean fields "
            f"{RR.ratio_txt(c2)}")
        if (s.variant == "DBs").any():
            est, ci_ = scoring.pooled_coverage(s, "DBs", "field"); c3 = scoring.contrast(s, "J1pK", "DBs", "field")
            say(f"    stratified estimator, field means: coverage {est:.3f} [{ci_[0]:.3f}, {ci_[1]:.3f}] ({int((s.variant == 'DBs').sum())} "
                f"blocks); CRPS TabPFN-GP / stratified estimator {RR.ratio_txt(c3)}")
    seeds = sorted(Path(out).glob("seeds_*.csv"))
    if seeds:
        s = pd.concat([pd.read_csv(f) for f in seeds], ignore_index=True)
        base = reg[reg.variant == "J1p"][KEY + ["M", "V"]].assign(seed=0)
        s = pd.concat([s[KEY + ["M", "V", "seed", "obs"]], base.merge(s[KEY + ["obs"]].drop_duplicates(), on=KEY)], ignore_index=True)
        h = F.Z80 * F.KAPPA_OURS * np.sqrt(s.V); s = s.assign(lo=s.M - h, hi=s.M + h, half=h)
        g = s.groupby(KEY).agg(lo_sd=("lo", "std"), hi_sd=("hi", "std"), half=("half", "mean"), k=("seed", "nunique")).reset_index()
        g = g[g.k == 5]; g["rel"] = np.maximum(g.lo_sd, g.hi_sd) / g.half
        for n_lab, dsg in (("n = 5", "n5"), ("n = 20", "n20")):
            q = g[g.design == dsg]
            say(f"  seeds, {n_lab}: {len(q)} blocks; sd of the interval endpoints over five seeds / half-width: median "
                f"{q.rel.median():.3f}, 90th percentile {q.rel.quantile(0.9):.3f}")
        cov = s.assign(c=(np.abs(s.obs - s.M) <= F.Z80 * F.KAPPA_OURS * np.sqrt(s.V)).astype(float)).groupby(["seed", "design"]).c.mean()
        say("  coverage by seed (n5 | n20): " + "; ".join(f"seed {k}: {cov.get((k, 'n5'), np.nan):.3f} | {cov.get((k, 'n20'), np.nan):.3f}"
                                                         for k in range(5)))
    (Path(out) / "report_designs.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["points", "rows", "seeds", "report"]); ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="*"); ap.add_argument("--workers", type=int, default=14); ap.add_argument("--final-rows")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage == "points":
        stage_points(a.out, a.fields)
    elif a.stage == "rows":
        stage_rows(a.out, a.workers)
    elif a.stage == "seeds":
        stage_seeds(a.out, a.fields)
    else:
        report(a.out, a.final_rows)


if __name__ == "__main__":
    main()
