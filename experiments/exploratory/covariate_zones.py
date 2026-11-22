"""E22 of docs/exploratory/plan_E22.md: zones formed from covariates on the 43 fields (exploratory).

Each field gets four zones from its covariates: k-means (4 clusters, random seed 0, best of ten starts) of the principal components that
carry 90 % of the variance of the field's standardised covariates (coordinates excluded), at all its locations, the rule of the sensitivity
check's zone set c4 (sensitivity.covariate_scores). The study's splits and stored point outputs are used; only the zones change. The zones
of block_cases.blocks_of are replaced and the study's own functions compute every method (block_cases.case_rows,
further_models.case_rows, further_models_rows.zone_block for the zone-effect model). Block kriging and WH run with the fixed variograms
only, the study's versions; the local-variogram variants, which no method of the study uses, are not computed. No existing module is changed.

  python -m experiments.exploratory.covariate_zones zones --out DIR                   (CPU)    zones and check 2 -> DIR/zones/, DIR/check2.txt
  python -m experiments.exploratory.covariate_zones check --out DIR --study-rows ST   (CPU, R) check 1 -> DIR/check1.txt
  python -m experiments.exploratory.covariate_zones rows --out DIR [--workers 12]     (CPU, R) DIR/rows/rows_e22_<field>_<design>.csv
  python -m experiments.exploratory.covariate_zones report --out DIR --study-rows ST --variants VA   -> DIR/report12.txt, DIR/report12.csv
Run from the repository root with SOILGP_DATA and SOILGP_RUN set to the study's data and run folders. ST: the study's stored rows
(rows_study_<field>.csv; models_study_<field>.csv or revision_study_<field>.csv); VA: the stored rows of the further models (variants_study_<field>.csv, zone-effect model).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one thread, as in the study's runs
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import json                                                                   # noqa: E402
import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from sklearn.cluster import KMeans                                            # noqa: E402
from sklearn.metrics import adjusted_rand_score                               # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.study import run as R                                        # noqa: E402
from experiments.paths import CONFIGS, REPO                                   # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402

SET = "e22"
CHECK_FIELDS = ("DK_FAA", "PO_NE", "DR_AUZ")
KEEP = ("J1p", "TAB", "KpR4", "RK_ins_fix_v", "WH_fix", "DB")                  # block_cases.case_rows: the study's methods
FURTHER = json.loads((REPO / "expected" / "exploratory" / "factors_once.json").read_text())
FACTOR_DEZ, FACTOR_Z, CRPS_Z = FURTHER["factor"]["DEz"], FURTHER["factor"]["Z"], FURTHER["crps_factor"]["Z"]
PUBLIC = {"NI_MAR", "NX_CZK", "NX_CZN", "NX_CZP", "NX_CZU", "NX_IT1", "NX_IT2", "NX_IT3", "NX_CHHB", "NX_CHZO", "DR_AUZ", "DR_EPO", "DR_PLO",
          "DR_CRO", "DR_F013", "DR_F111", "DR_WAL", "DR_HEN"} | {f for f in R.CODES if f.startswith("PO_")}   # as further_models_report.py
NAMES = {"J1p": "TabPFN-GP", "O1": "Shared-error-only", "TAB": "TabPFN", "KpR4": "TabPFN-KpR", "RK_ins_fix_v": "Block RK", "WH_fix": "WH",
         "DB": "Sample mean", "DEz": "Difference estimator", "Z": "Zone-effect model"}
COMPS = ["TAB", "KpR4", "RK_ins_fix_v", "WH_fix"]
METHODS = ["J1p", "O1"] + COMPS + ["DB", "DEz", "Z"]
LABELS = {}                                                                   # field -> zone label of every location (this process)
lines = []


def say(s=""):
    lines.append(s); print(s, flush=True)


# ------------------------------------------------------------------------------------------------------------------------- zones
def covariate_scores(data):
    """Principal components (90 % of the variance) of the field's standardised covariates at all its locations (sensitivity.covariate_scores;
    the study's covariates hold no spectra and no missing values, checked in check 2)."""
    X = np.asarray(data["X"], float); X = X[:, np.nanstd(X, 0) > 0]
    assert np.all(np.isfinite(X)), "non-finite covariates"
    Z = (X - X.mean(0)) / X.std(0); U, s, _ = np.linalg.svd(Z, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.90) + 1)
    return U[:, :k] * s[:k]


def labels_of(data, kind):
    km = lambda X: KMeans(S_.ZONE_K, random_state=0, n_init=10).fit_predict(X)          # noqa: E731
    return km(np.asarray(data["coordinates"], float)) if kind == "compact" else km(covariate_scores(data))


def blocks_of(case):
    """block_cases.blocks_of with the zones of LABELS: field and zone blocks (indices into te; a zone where it holds at least 3 unrevealed
    locations) and, per block, the revealed rows in the same area."""
    lab = LABELS[case["field"]]; tr, te = case["tr"], case["te"]
    blocks = {"field": np.arange(len(te))}
    for j in range(int(lab.max()) + 1):
        idx = np.flatnonzero(lab[te] == j)
        if len(idx) >= S_.ZONE_MIN_POINTS:
            blocks[f"zone{j}"] = idx
    revealed = {b: (tr if b == "field" else tr[lab[tr] == int(b[4:])]) for b in blocks}
    return blocks, revealed


def install(field, labels):
    """Use the given zones in the study's code (this process only); skip the local-variogram R part, which no method of the study uses."""
    from experiments.core import block_cases
    LABELS[field] = np.asarray(labels, int)
    if block_cases.blocks_of is not blocks_of:
        block_cases.blocks_of = blocks_of
        run_r = block_cases.run_r

        def fixed_only(mode, S, tr, te, columns, params=None):
            if params is not None and params.get("parts") == ["loc"]:
                return {}, {}
            return run_r(mode, S, tr, te, columns, params)
        block_cases.run_r = fixed_only
    return block_cases


def zone_effect():
    """further_models_rows.zone_block (that module reads its arguments when imported)."""
    argv = sys.argv; sys.argv = ["further_models_rows", "study", os.devnull]
    try:
        from experiments.exploratory import further_models_rows as FMR
    finally:
        sys.argv = argv
    return FMR.zone_block


def z_rows(case, blocks, revealed, labels, zone_block):
    """Zone-effect model rows, combined with TabPFN as in further_models_rows.case_rows."""
    S, tr, te, y = case["S"], case["tr"], case["te"], case["y"]; n = len(tr); yq = y[te]; mT, sT = case["mT"], case["sT"]
    key = dict(set=SET, src="external" if case["field"] in F.C.EXTERNAL else "limesoda", field=case["field"], design=case["design"],
               rep=case["rep"], target=case["target"], n=n, m=len(te))
    rows = []

    def add(b, M, V, err=""):
        i = blocks[b]; failed = bool(err) or not (np.isfinite(M) and np.isfinite(V) and V > 1e-18)
        rows.append(dict(key, block=b, mB=len(i), nB=len(revealed[b]), variant="Z", M=float(M), V=float(V), df=np.inf,
                         obs=float(yq[i].mean()), failed=failed, own_fallback=False, err=err))
    try:
        mom, vbar = zone_block(labels[tr], labels[te], y[tr] - case["m_oof"], blocks)
        ind = np.clip(sT ** 2 - vbar, 0, None)
        for b, i in blocks.items():
            mb, vb = mom[b]
            add(b, float(mT[i].mean()) + (mb if n >= F.GATE_N else 0.0), float(np.sum(ind[i])) / len(i) ** 2 + vb)
    except Exception as e:                                                    # counted and reported, never replaced
        for b in blocks:
            add(b, np.nan, np.nan, err=repr(e)[:150])
    return rows


def case_rows(case, labels, shapes, zone_block):
    """Every method's rows for one case with the zones of `labels` (installed for the case's field)."""
    from experiments.core import block_cases
    from experiments.pipeline import further_models
    rows = [x for x in block_cases.case_rows(case, shapes, ours=True) if x["variant"] in KEEP]
    rows += [x for x in further_models.case_rows(case)[0] if x["variant"] == "O1" or (x["variant"] == "DEz" and x["block"] != "field")]
    if case["m_oof"] is not None:
        blocks, revealed = block_cases.blocks_of(case)
        rows += z_rows(case, blocks, revealed, labels, zone_block)
    for x in rows:
        x["set"] = SET
    return rows


def save_zones(out, field, comp, cov):
    p = Path(out) / "zones" / f"{field}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"compact": comp.tolist(), "covariate": cov.tolist()}))


def load_zones(out, field):
    d = json.loads((Path(out) / "zones" / f"{field}.json").read_text())
    return np.asarray(d["compact"], int), np.asarray(d["covariate"], int)


def between_share(y, lab):
    """Share of the variance of y between the zones (between-zone sum of squares over the total)."""
    tot = float(np.sum((y - y.mean()) ** 2))
    return float(sum(np.sum(lab == j) * (y[lab == j].mean() - y.mean()) ** 2 for j in np.unique(lab))) / tot if tot > 0 else np.nan


def stage_zones(out, fields):
    """The zones of every field and check 2."""
    from experiments.core.common import load_field
    say("=== E22 zones and check 2 (docs/exploratory/plan_E22.md) ===")
    say("field: covariates used / in the file; components (90 %); covariate-zone sizes (all locations); adjusted Rand index with the compact "
        "zones; per property the share of its variance between zones, covariate | compact")
    desc = []
    for f in fields:
        data = load_field(f); X = np.asarray(data["X"], float); cols = [str(c) for c in data["columns"]]
        spectral = sum(c.startswith("wl_") for c in cols); finite = bool(np.isfinite(X).all()); used = int((np.nanstd(X, 0) > 0).sum())
        if not finite or used == 0 or spectral:
            say(f"  {f}: LEFT OUT ({'missing covariate values' if not finite else 'spectral columns' if spectral else 'no covariate varies'})")
            continue
        comp, cov = labels_of(data, "compact"), labels_of(data, "covariate"); save_zones(out, f, comp, cov)
        k = covariate_scores(data).shape[1]; sizes = np.bincount(cov, minlength=S_.ZONE_K); ari = adjusted_rand_score(comp, cov)
        shares = []
        for t in R.tasks(f):
            y = data["Y"][:, F.C.TARGETS.index(t)]
            shares.append(f"{t.replace('_target', '')} {between_share(y, cov):.2f} | {between_share(y, comp):.2f}")
        small = [int(j) for j in range(S_.ZONE_K) if sizes[j] < S_.ZONE_MIN_POINTS]
        say(f"  {f}: {used}/{len(cols)} covariates; {k} components; sizes {'/'.join(map(str, sizes))}; ARI {ari:.2f}; " + "; ".join(shares)
            + (f"; zone(s) {small} never evaluated (fewer than {S_.ZONE_MIN_POINTS} locations)" if small else ""))
        desc.append(dict(field=f, covariates=used, components=k, sizes="/".join(map(str, sizes)), ari=ari, min_size=int(sizes.min())))
    d = pd.DataFrame(desc)
    say(f"\n{len(d)} fields; components median {d.components.median():.0f} (range {d.components.min()}-{d.components.max()}); "
        f"adjusted Rand index median {d.ari.median():.2f} (range {d.ari.min():.2f}-{d.ari.max():.2f}); smallest covariate zone median "
        f"{d.min_size.median():.0f} locations (range {d.min_size.min()}-{d.min_size.max()})")
    d.to_csv(Path(out) / "zones" / "zones.csv", index=False)
    (Path(out) / "check2.txt").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------------------------------------- rows
def job_rows(job):
    """Rows of one field and design (all repeats and properties) with the covariate zones."""
    out, field, design = job
    from experiments.core.common import load_field
    R.setup(R.CODES); shapes = json.loads((CONFIGS / "fixed" / "shapes.json").read_text())
    _, cov = load_zones(out, field); block_cases = install(field, cov); zone_block = zone_effect()
    data = load_field(field); rows, missing, t0 = [], 0, time.monotonic()
    for f, data, d, rep, tr in block_cases.iter_splits("study", [field]):
        if d != design:
            continue
        for target in R.tasks(field):
            case = block_cases.load_case("study", field, data, d, rep, tr, target)
            if case is None:
                missing += 1; continue
            rows += case_rows(case, cov, shapes, zone_block)
    p = Path(out) / "rows"; p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p / f"rows_{SET}_{field}_{design}.csv", index=False)
    return field, design, len(rows), missing, time.monotonic() - t0


def stage_rows(out, fields, workers):
    from experiments.core.common import load_field
    jobs = []
    for f in fields:
        if not (Path(out) / "zones" / f"{f}.json").exists():
            continue
        N = len(load_field(f)["coordinates"])
        for d in F.C.N_DESIGNS:
            if N - int(d[1:]) >= F.MIN_TEST and not (Path(out) / "rows" / f"rows_{SET}_{f}_{d}.csv").exists():
                jobs.append((out, f, d))
    jobs.sort(key=lambda j: -int(j[2][1:]) * len(load_field(j[1])["coordinates"]))              # the largest first
    print(f"{len(jobs)} field x design jobs", flush=True)
    with Pool(min(workers, len(jobs))) as pool:
        for field, design, n, missing, sec in pool.imap_unordered(job_rows, jobs):
            print(f"rows {field} {design}: {n} rows, {missing} cases without point outputs, {sec:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- check 1
def stored_rows(study_rows, variants, fields=None):
    """The study's stored rows of every method (compact zones)."""
    parts = []
    for f in sorted(Path(study_rows).glob("rows_study_*.csv")):
        q = pd.read_csv(f, low_memory=False); parts.append(q[q.variant.isin(KEEP)])
    for pat in ("models_study_*.csv", "revision_study_*.csv"):                # the shared-error-only rows, under either name
        for f in sorted(Path(study_rows).glob(pat)):
            q = pd.read_csv(f, low_memory=False); parts.append(q[(q.variant == "O1") | ((q.variant == "DEz") & (q.block != "field"))])
    for f in sorted(Path(variants).glob("variants_study_*.csv")):
        q = pd.read_csv(f, low_memory=False); parts.append(q[q.variant == "Z"])
    r = pd.concat(parts, ignore_index=True); r["failed"] = r.failed.astype(bool)
    return r[r.field.isin(fields)] if fields else r


def stage_check(out, study_rows, variants):
    """Check 1: with the compact zones installed, this module reproduces the stored rows of every method."""
    from experiments.core.common import load_field, training_rows
    R.setup(R.CODES); shapes = json.loads((CONFIGS / "fixed" / "shapes.json").read_text()); zone_block = zone_effect()
    rows = []
    for field in CHECK_FIELDS:
        data = load_field(field); comp = labels_of(data, "compact"); block_cases = install(field, comp)
        for rep in (300, 301):
            tr = training_rows(data, field, "n20", rep)
            for target in R.tasks(field):
                case = block_cases.load_case("study", field, data, "n20", rep, tr, target)
                rows += case_rows(case, comp, shapes, zone_block)
    new = pd.DataFrame(rows)
    old = stored_rows(study_rows, variants, list(CHECK_FIELDS)); old = old[(old.design == "n20") & old.rep.isin([300, 301])]
    say("=== E22 check 1: the compact zones through this module against the stored rows ===")
    worst = 0.0
    for v in METHODS:
        a = new[new.variant == v].set_index(R.KEYS)[["M", "V"]]; b = old[old.variant == v].set_index(R.KEYS)[["M", "V"]]
        j = a.join(b, lsuffix="_new", rsuffix="_old", how="inner")
        rel = max(float((np.abs(j.M_new - j.M_old) / np.maximum(np.abs(j.M_old), 1e-12)).max()),
                  float((np.abs(j.V_new - j.V_old) / np.maximum(np.abs(j.V_old), 1e-12)).max())) if len(j) else np.inf
        complete = len(j) == len(b) == len(a)
        worst = max(worst, rel if complete else np.inf)
        say(f"  {NAMES[v]:22s} {len(j)} of {len(b)} stored blocks matched ({len(a)} new); largest relative difference in M or V {rel:.2e}")
    say(f"check 1 {'passed' if worst < 1e-6 else 'FAILED: stop'} (largest {worst:.2e})")
    (Path(out) / "check1.txt").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------------------------------------- report
def scored(raw):
    from experiments.core import scoring
    _, factors, crps_f, _ = R.methods()
    factors = {**factors, "DEz": FACTOR_DEZ, "Z": FACTOR_Z}; crps_f = {**crps_f, "DEz": 1.0, "Z": CRPS_Z}
    sc = scoring.score(raw, factors); sc["crps"] = scoring.score(raw, crps_f)["crps"].to_numpy()
    return sc, factors, crps_f


def method_set(s, v):
    """The blocks a method is scored on, as in the study: TabPFN-GP and the four comparators where all five are scored; the
    shared-error-only and zone-effect models where TabPFN-GP is also scored; the sample mean and the difference estimator on their own."""
    from experiments.core import scoring
    if v in ["J1p"] + COMPS:
        q = scoring.common_blocks(s, ["J1p"] + COMPS)
    elif v in ("O1", "Z"):
        q = scoring.common_blocks(s, ["J1p", v])
    else:
        q = s
    return q[(q.variant == v) & (q.block != "field") & q.cov80.notna()]


def per_field(q):
    return q.groupby("field").cov80.agg(["sum", "count"])


def paired_change(a, b):
    ga, gb = per_field(a), per_field(b); f = ga.index.intersection(gb.index)
    if len(f) == 0:
        return np.nan, (np.nan, np.nan)
    ga, gb = ga.loc[f], gb.loc[f]; idx = R.boot_idx(len(f))
    est = ga["sum"].sum() / ga["count"].sum() - gb["sum"].sum() / gb["count"].sum()
    bs = ga["sum"].to_numpy()[idx].sum(1) / ga["count"].to_numpy()[idx].sum(1) - gb["sum"].to_numpy()[idx].sum(1) / gb["count"].to_numpy()[idx].sum(1)
    return float(est), tuple(np.percentile(bs, [2.5, 97.5]))


def verdict(est, ci):
    lo, hi = R.BAND
    return ("keeps its coverage" if lo <= est <= hi else "too few" if ci[1] < lo else "too many" if ci[0] > hi else "given as it is")


csv_rows = []


def stage_report(out, study_rows, variants):
    from experiments.core import scoring
    R.setup(R.CODES)
    fmt = lambda e, ci: f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"                  # noqa: E731
    new = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted((Path(out) / "rows").glob(f"rows_{SET}_*.csv"))], ignore_index=True)
    new["failed"] = new.failed.astype(bool)
    fields = sorted(new.field.unique()); old = stored_rows(study_rows, variants, fields)
    sc_new, factors, crps_f = scored(new); sc_old, _, _ = scored(old)
    sets = {"covariate": sc_new, "compact": sc_old}
    for name in ("check1.txt", "check2.txt"):                                # checks 1 and 2, as written by their stages
        p = Path(out) / name
        if p.exists():
            say(p.read_text().rstrip() + "\n")
    say("=== E22: zones formed from covariates on the 43 fields (docs/exploratory/plan_E22.md; exploratory) ===")
    say("80 % intervals with each method's study factor (" + ", ".join(f"{NAMES[v]} {factors[v]:.2f}" for v in METHODS) + "); "
        "[95 % interval over fields] (fields); change from the compact zones on the same splits, paired by field")

    # field rows: the field block is the study's, so every method but the zone-effect model must reproduce the stored rows
    a = new[(new.block == "field") & (new.variant != "Z")].set_index(R.KEYS + ["variant"])[["M", "V"]]
    b = old[(old.block == "field") & (old.variant != "Z")].set_index(R.KEYS + ["variant"])[["M", "V"]]
    j = a.join(b, lsuffix="_new", rsuffix="_old", how="inner")
    rel = max(float((np.abs(j.M_new - j.M_old) / np.maximum(np.abs(j.M_old), 1e-12)).max()),
              float((np.abs(j.V_new - j.V_old) / np.maximum(np.abs(j.V_old), 1e-12)).max()))
    say(f"field rows against the stored ones: {len(j)} of {len(b)} matched ({len(a)} new); largest relative difference in M or V {rel:.2e}")
    say("zone means scored: " + "; ".join(f"{k}: " + ", ".join(f"{NAMES[v]} {len(method_set(s, v))}" for v in METHODS) for k, s in sets.items()))
    say("failed zone predictions (covariate zones): " + ", ".join(
        f"{NAMES[v]} {int(new[(new.variant == v) & (new.block != 'field')].failed.sum())}/{int(((new.variant == v) & (new.block != 'field')).sum())}"
        for v in METHODS))

    say("\n-- check 3: the sample mean's coverage of covariate-zone means (theory: near 0.80 under simple random sampling) --")
    e, ci, k = R.coverage(method_set(sc_new, "DB"), "DB", "zone"); e0, ci0, _ = R.coverage(method_set(sc_old, "DB"), "DB", "zone")
    ok = e >= R.BAND[0]
    say(f"  covariate zones {fmt(e, ci)} ({k} fields); compact zones {fmt(e0, ci0)} -> {'passed' if ok else 'FAILED: look for an error'}")
    if not ok:
        (Path(out) / "report12.txt").write_text("\n".join(lines) + "\n"); sys.exit(2)

    say("\n-- coverage of zone means: covariate zones | compact zones | change; reading (band [0.75, 0.85]) --")
    for v in METHODS:
        A, B = method_set(sc_new, v), method_set(sc_old, v)
        e, ci, k = R.coverage(A, v, "zone"); e0, ci0, _ = R.coverage(B, v, "zone"); d, dci = paired_change(A, B)
        say(f"  {NAMES[v]:22s} {fmt(e, ci)} ({k}) | {fmt(e0, ci0)} | {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}] -> {verdict(e, ci)}")
        csv_rows.append(dict(section="coverage", method=NAMES[v], variant=v, subset="all", coverage=e, lo=ci[0], hi=ci[1], fields=k,
                             compact=e0, compact_lo=ci0[0], compact_hi=ci0[1], change=d, change_lo=dci[0], change_hi=dci[1], reading=verdict(e, ci)))

    say("\n-- by the zone's number of revealed samples: covariate | compact | change --")
    for lab, sel in (("none", lambda q: q.nB == 0), ("one or two", lambda q: q.nB.isin([1, 2])), ("three or more", lambda q: q.nB >= 3)):
        say(f"  zones with {lab} revealed samples")
        for v in METHODS:
            A, B = method_set(sc_new, v), method_set(sc_old, v); A, B = A[sel(A)], B[sel(B)]
            e, ci, k = R.coverage(A, v, "zone")
            if not np.isfinite(e):
                continue
            e0, ci0, _ = R.coverage(B, v, "zone"); d, dci = paired_change(A, B)
            say(f"    {NAMES[v]:22s} {fmt(e, ci)} ({k}; {len(A)} means) | {fmt(e0, ci0)} | {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")
            csv_rows.append(dict(section="revealed", method=NAMES[v], variant=v, subset=lab, coverage=e, lo=ci[0], hi=ci[1], fields=k,
                                 compact=e0, compact_lo=ci0[0], compact_hi=ci0[1], change=d, change_lo=dci[0], change_hi=dci[1]))

    say("\n-- by group of fields, by covariate source and by n: covariate | compact | change --")
    subsets = [(g, [f for f in fields if R.group(f) == g]) for g in ("Uniformity trials", "SensRes", "Midwest", "New fields", "Further fields")]
    subsets += [("released covariates", [f for f in fields if f not in PUBLIC]), ("public covariates", [f for f in fields if f in PUBLIC])]
    for lab, fl in subsets:
        say(f"  {lab} ({len(fl)} fields)")
        for v in METHODS:
            A, B = method_set(sc_new, v), method_set(sc_old, v); A, B = A[A.field.isin(fl)], B[B.field.isin(fl)]
            e, ci, k = R.coverage(A, v, "zone")
            if not np.isfinite(e):
                continue
            e0, ci0, _ = R.coverage(B, v, "zone"); d, dci = paired_change(A, B)
            say(f"    {NAMES[v]:22s} {fmt(e, ci)} ({k}) | {fmt(e0, ci0)} | {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")
            csv_rows.append(dict(section="subset", method=NAMES[v], variant=v, subset=lab, coverage=e, lo=ci[0], hi=ci[1], fields=k,
                                 compact=e0, compact_lo=ci0[0], compact_hi=ci0[1], change=d, change_lo=dci[0], change_hi=dci[1]))
    ns = sorted(new.n.unique())
    for v in METHODS:
        parts = []
        for n in ns:
            A, B = method_set(sc_new, v), method_set(sc_old, v); A, B = A[A.n == n], B[B.n == n]
            e, ci, k = R.coverage(A, v, "zone")
            if np.isfinite(e):
                e0 = R.coverage(B, v, "zone")[0]; parts.append(f"{n}: {e:.3f} | {e0:.3f}")
                csv_rows.append(dict(section="by_n", method=NAMES[v], variant=v, subset=f"n={n}", coverage=e, lo=ci[0], hi=ci[1], fields=k, compact=e0))
        say(f"  {NAMES[v]:22s} by n " + "; ".join(parts))

    say("\n-- rise in coverage D of zone means (n >= 30 minus n <= 10): covariate zones [90 %] [95 %] | compact zones --")
    for v in METHODS:
        out_ = []
        for k_, s in sets.items():
            q = method_set(s, v); est, bs, nf = R.rise(q, v, "zone")
            out_.append((est, np.percentile(bs, [5, 95]), np.percentile(bs, [2.5, 97.5]), nf))
        (e, c90, c95, nf), (e0, _, c950, _) = out_
        say(f"  {NAMES[v]:22s} D = {e:+.3f} [{c90[0]:+.3f}, {c90[1]:+.3f}] [{c95[0]:+.3f}, {c95[1]:+.3f}] ({nf}) | {e0:+.3f} "
            f"[{c950[0]:+.3f}, {c950[1]:+.3f}]")
        csv_rows.append(dict(section="rise", method=NAMES[v], variant=v, subset="D", coverage=e, lo=c95[0], hi=c95[1], lo90=c90[0], hi90=c90[1],
                             fields=nf, compact=e0, compact_lo=c950[0], compact_hi=c950[1]))

    say("\n-- CRPS and interval-width ratio TabPFN-GP / method for zone means: covariate zones | compact zones --")
    for v in METHODS[1:]:
        for col in ("crps", "width"):
            res = []
            for k_, s in sets.items():                                    # the study's sets: comparators where all five are scored,
                common = scoring.common_blocks(s, ["J1p"] + COMPS)            # design-based estimators against TabPFN-GP on that set,
                q = (common if v in COMPS else s if v in ("O1", "Z")          # the models wherever both are scored
                     else pd.concat([common[common.variant == "J1p"], s[s.variant == v]]))
                res.append(scoring.contrast(q, "J1p", v, "zone", col=col))
            c, c0 = res
            if c is None:
                continue
            say(f"  {col:5s} TabPFN-GP / {NAMES[v]:22s} {fmt(c['ratio'], c['ci'])}, lower on {c['better']} of {c['fields']} fields | "
                f"{fmt(c0['ratio'], c0['ci']) if c0 else '-'}")
            csv_rows.append(dict(section=f"{col}_ratio", method=NAMES[v], variant=v, subset="all", coverage=c["ratio"], lo=c["ci"][0],
                                 hi=c["ci"][1], fields=c["fields"], better=c["better"], compact=c0["ratio"] if c0 else np.nan,
                                 compact_lo=c0["ci"][0] if c0 else np.nan, compact_hi=c0["ci"][1] if c0 else np.nan))

    say("\n-- coverage by field, TabPFN-GP and the sample mean: covariate | compact --")
    for f in fields:
        a1, b1 = method_set(sc_new, "J1p"), method_set(sc_old, "J1p"); a2, b2 = method_set(sc_new, "DB"), method_set(sc_old, "DB")
        say(f"  {f:8s} TabPFN-GP {R.coverage(a1, 'J1p', 'zone', [f])[0]:.3f} | {R.coverage(b1, 'J1p', 'zone', [f])[0]:.3f}; "
            f"sample mean {R.coverage(a2, 'DB', 'zone', [f])[0]:.3f} | {R.coverage(b2, 'DB', 'zone', [f])[0]:.3f}; "
            f"zone means {len(a1[a1.field == f])} | {len(b1[b1.field == f])}")
    (Path(out) / "report12.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(Path(out) / "report12.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["zones", "check", "rows", "report"])
    ap.add_argument("--out", required=True); ap.add_argument("--fields", nargs="*"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--study-rows"); ap.add_argument("--variants")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    R.setup(R.CODES); fields = list(a.fields or R.evaluated())
    if a.stage == "zones":
        stage_zones(a.out, fields)
    elif a.stage == "check":
        stage_check(a.out, a.study_rows, a.variants)
    elif a.stage == "rows":
        stage_rows(a.out, fields, a.workers)
    else:
        stage_report(a.out, a.study_rows, a.variants)


if __name__ == "__main__":
    main()
