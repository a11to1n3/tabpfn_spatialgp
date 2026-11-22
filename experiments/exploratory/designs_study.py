"""E21 of docs/exploratory/plan_E21.md: other sampling designs on the 43 fields (exploratory).

Each design chooses which of a field's measured locations are revealed; the rest are the targets, as in the study. The splits are passed
to the study's own code: point_model.our_model (TabPFN and its cross-fitted residuals), block_cases.case_rows and further_models.case_rows
(the area rows of every method), scoring (the scores). No existing module is changed.

  python -m experiments.exploratory.designs_study splits --out DIR                  (CPU)   splits, checks 2-4 -> DIR/splits/, DIR/checks.txt
  python -m experiments.exploratory.designs_study check --out DIR --study-rows ST   (GPU)   check 1 on the study's own splits
  python -m experiments.exploratory.designs_study points --out DIR [--fields ...]   (GPU)   DIR/points/ours/<design>/<field>/<target>/rep<r>.npz
  python -m experiments.exploratory.designs_study rows --out DIR [--workers 12]     (CPU, R) DIR/rows/rows_e21_<field>.csv
  python -m experiments.exploratory.designs_study report --out DIR --study-rows ST  ->      DIR/report11.txt, DIR/report11.csv
Run from the repository root with SOILGP_DATA set to the study's data folder. ST: the folder of the study's stored rows (rows_study_<field>.csv
and models_study_<field>.csv, or revision_study_<field>.csv for the shared-error-only rows).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one thread, as in the study's runs
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import json                                                                   # noqa: E402
import time                                                                   # noqa: E402
import zlib                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from scipy.spatial import cKDTree                                             # noqa: E402
from sklearn.cluster import KMeans                                            # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.study import run as R                                        # noqa: E402
from experiments.paths import CONFIGS, REPO                                   # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402

SET = "e21"
SIZES = (10, 20, 30)
REPS = range(400, 410)
FIXED = ("strat", "clust", "tcov", "tsoil")
SPACINGS = (2, 3)
GAMMA = 1.5
MIN_GRID_N = 5
CHECK_FIELDS = ("DK_FAA", "PO_NE", "DR_AUZ")
FACTOR_DE = 1.05                                                              # the difference estimators' factor (table of further methods)
lines = []


def say(s=""):
    lines.append(s); print(s, flush=True)


def regular_fields():
    t = pd.read_csv(REPO / "expected" / "study" / "field_areas.csv")
    return tuple(t.field[t.layout == "regular"])


def rng_of(field, design, rep, target=None):
    key = f"{field}|{design}|{target}|{rep}" if target else f"{field}|{design}|{rep}"
    return np.random.default_rng(zlib.crc32(key.encode()))


def zone_labels(S):
    return KMeans(S_.ZONE_K, random_state=0, n_init=10).fit_predict(S)       # the clustering of zone_blocks


# ------------------------------------------------------------------------------------------------------------------------- designs
def lattice(S):
    """Lattice index (i, j) of each location on a regular layout, for a lattice of any shape (square, rectangular, staggered,
    triangular): the first direction is that of the nearest-neighbour vectors, the second the shortest neighbour vector at more than
    20 degrees from it; each location is written as i * b1 + j * b2 from the lattice's offset. Returns (i, j, share of locations within
    a quarter of a lattice step of their node, all nodes distinct)."""
    S = np.asarray(S, float); t = cKDTree(S); _, nb = t.query(S, k=min(9, len(S)))
    V = (S[nb[:, 1:]] - S[:, None, :]).reshape(-1, 2)
    V = np.where((V[:, 1] < 0) | ((V[:, 1] == 0) & (V[:, 0] < 0)), -1, 1)[:, None] * V        # one sign per direction
    L = np.linalg.norm(V, axis=1); A = np.degrees(np.arctan2(V[:, 1], V[:, 0]))              # angles in [0, 180)
    h1 = np.median(np.linalg.norm(S[nb[:, 1]] - S, axis=1))
    def mode_dir(mask):
        a = A[mask]; hist, edges = np.histogram(a, bins=36, range=(0, 180)); k = int(np.argmax(hist))
        sel = mask & (A >= edges[k] - 5) & (A < edges[k + 1] + 5)
        return np.median(V[sel], axis=0)
    b1 = mode_dir(L <= 1.15 * h1)
    a1 = np.degrees(np.arctan2(b1[1], b1[0])) % 180
    diff = np.abs(((A - a1) + 90) % 180 - 90)
    other = diff > 20
    if not other.any():
        return np.zeros(len(S), int), np.zeros(len(S), int), 0.0, False
    h2 = np.median(np.sort(L[other])[: max(1, int(0.25 * other.sum()))])
    b2 = mode_dir(other & (L <= 1.15 * h2))
    C = (S - S.mean(0)) @ np.linalg.inv(np.column_stack([b1, b2])).T                          # coordinates in lattice steps
    out = []
    for col in (0, 1):
        x = C[:, col]; phase = np.angle(np.mean(np.exp(2j * np.pi * x))) / (2 * np.pi)
        k = np.rint(x - phase).astype(int); out.append((k, np.abs(x - phase - k)))
    (i, ri), (j, rj) = out
    ok = float(np.mean((ri < 0.25) & (rj < 0.25)))
    return i, j, ok, len(set(zip(i.tolist(), j.tolist()))) == len(S)


def grid_samples(S, k):
    """The k*k grid samples at spacing k: every k-th location along both axes, for each start (a, b); rep = a * k + b."""
    i, j, _, _ = lattice(S)
    return {a * k + b: np.flatnonzero((i % k == a) & (j % k == b)) for a in range(k) for b in range(k)}


def pc1(data):
    X = np.asarray(data["X"], float); sd = X.std(0); X = X[:, sd > 0]
    Z = (X - X.mean(0)) / X.std(0)
    s = Z @ np.linalg.svd(Z, full_matrices=False)[2][0]
    return s / s.std()


def fixed_design(data, field, design, n, rep, target=None):
    """Revealed rows of one split of strat, clust, tcov or tsoil with n locations; ValueError if infeasible. Also the signed z for tcov,
    tsoil (for check 2)."""
    S = np.asarray(data["coordinates"], float); N = len(S)
    if N - n < F.MIN_TEST:
        raise ValueError(f"{design}{n} leaves fewer than {F.MIN_TEST} unrevealed locations")
    rng = rng_of(field, f"{design}{n}", rep, target if design == "tsoil" else None)
    if design == "strat":
        z = zone_labels(S); alloc = np.full(S_.ZONE_K, n // S_.ZONE_K)
        alloc[rng.choice(S_.ZONE_K, n - alloc.sum(), replace=False)] += 1
        out = []
        for h in range(S_.ZONE_K):
            idx = np.flatnonzero(z == h)
            if len(idx) < alloc[h]:
                raise ValueError(f"zone {h} holds {len(idx)} locations, fewer than its {alloc[h]}")
            out += list(rng.choice(idx, alloc[h], replace=False))
        return np.sort(np.array(out)), None
    if design == "clust":                                                     # as the cluster designs of core/common.py, with n locations
        seeds = rng.choice(N, 4, replace=False); chosen = set(seeds.tolist())
        order = [np.argsort(np.linalg.norm(S - S[s], axis=1)) for s in seeds]; ptr = [0] * 4; j = 0
        while len(chosen) < n:
            o = order[j % 4]
            while o[ptr[j % 4]] in chosen:
                ptr[j % 4] += 1
            chosen.add(int(o[ptr[j % 4]])); j += 1
        return np.array(sorted(chosen)), None
    if design in ("tcov", "tsoil"):
        if design == "tcov":
            z = pc1(data)
        else:
            y = data["Y"][:, F.C.TARGETS.index(target)]; z = (y - y.mean()) / y.std()
        sign = rng.choice([-1.0, 1.0]); w = np.exp(GAMMA * sign * z); tr = np.sort(rng.choice(N, n, replace=False, p=w / w.sum()))
        return tr, sign * z
    raise ValueError(design)


def field_splits(field):
    """Every split of one field: {(design, rep, target or ''): revealed rows}, and the skipped designs with their reasons."""
    from experiments.core.common import load_field
    data = load_field(field); S = np.asarray(data["coordinates"], float); N = len(S)
    tasks = R.tasks(field); splits, skipped, signed = {}, [], []
    for design in FIXED:
        for n in SIZES:
            for target in (tasks if design == "tsoil" else (None,)):
                for rep in REPS:
                    try:
                        tr, sz = fixed_design(data, field, design, n, rep, target)
                    except ValueError as e:
                        skipped.append((field, f"{design}{n}", target or "", str(e))); break
                    splits[(f"{design}{n}", rep, target or "")] = tr
                    if sz is not None:
                        signed.append(dict(field=field, design=design, n=n, target=target or "", rep=rep, zbar=float(sz[tr].mean())))
    lat = None
    if field in regular_fields():
        i, j, ok, distinct = lattice(S); lat = dict(field=field, ok=ok, distinct=distinct)
        if ok >= 0.95 and distinct:
            for k in SPACINGS:
                g = {r: tr for r, tr in grid_samples(S, k).items() if len(tr) >= MIN_GRID_N and N - len(tr) >= F.MIN_TEST}
                if not g:
                    skipped.append((field, f"grid{k}", "", "no grid sample with at least 5 locations")); continue
                for r, tr in g.items():
                    splits[(f"grid{k}", r, "")] = tr
                m = int(np.median([len(tr) for tr in g.values()]))
                for rep in REPS:
                    splits[(f"gsrs{k}", rep, "")] = np.sort(rng_of(field, f"gsrs{k}", rep).choice(N, m, replace=False))
        else:
            skipped.append((field, "grid", "", f"lattice not recovered (share on nodes {ok:.3f}, nodes distinct {distinct})"))
    return field, splits, skipped, signed, lat


def save_splits(out, field, splits):
    p = Path(out) / "splits" / f"{field}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([[d, r, t, tr.tolist()] for (d, r, t), tr in splits.items()]))


def load_splits(out, field):
    p = Path(out) / "splits" / f"{field}.json"
    return [(d, r, t, np.asarray(tr, int)) for d, r, t, tr in json.loads(p.read_text())]


def stage_splits(out, fields):
    from experiments.core.common import load_field
    say("=== E21 splits (docs/exploratory/plan_E21.md) ===")
    allsk, allz, lats, counts = [], [], [], {}
    for field in fields:
        f, sp, sk, sz, lat = field_splits(field); save_splits(out, f, sp)
        allsk += sk; allz += sz
        if lat:
            lats.append(lat)
        N = len(load_field(f)["coordinates"])
        for (d, r, t), tr in sp.items():                                     # check 2: n distinct locations, at least 10 unrevealed
            assert len(np.unique(tr)) == len(tr) and N - len(tr) >= F.MIN_TEST, (f, d, r, t)
            key = "".join(c for c in d if not c.isdigit()) if not d.startswith(("grid", "gsrs")) else d
            counts[key] = counts.get(key, 0) + (1 if t else len(R.tasks(f)))
        say(f"{f}: {len(sp)} splits")
    say("\nsplits x tasks by design: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"; total {sum(counts.values())}")
    say("\n-- check 3: lattice of the regular fields --")
    for l in lats:
        say(f"  {l['field']}: share on nodes {l['ok']:.3f}, nodes distinct {l['distinct']}")
    say("\n-- skipped (field, design, property, reason) --")
    for s in allsk:
        say("  " + " | ".join(map(str, s)))
    z = pd.DataFrame(allz)
    say("\n-- check 2: mean signed z at the revealed locations (tcov, tsoil), per field: minimum over the field's splits of the field mean --")
    if not z.empty:
        g = z.groupby(["design", "field"]).zbar.mean().reset_index()
        for d, q in g.groupby("design"):
            low = q[q.zbar < 0.5]
            say(f"  {d}: field means from {q.zbar.min():.2f} to {q.zbar.max():.2f}; fields below 0.5: {', '.join(low.field) or 'none'}")
    check4(out, fields)
    (Path(out) / "checks.txt").write_text("\n".join(lines) + "\n")


def sample_mean_rows(key, S, tr, te, y):
    """The sample mean of the field's revealed values, as in block_cases (t with n - 1 degrees of freedom)."""
    yr = y[tr]; m = len(te)
    return dict(key, block="field", mB=m, nB=len(tr), variant="DB", M=float(yr.mean()), V=float(yr.var(ddof=1)) * (1 / len(tr) + 1 / m),
                df=float(len(tr) - 1), obs=float(y[te].mean()), failed=False)


def check4(out, fields):
    """Check 4: the sample mean's coverage of field means under grid against gsrs, before any other result is read."""
    from experiments.core.common import load_field
    from experiments.core import scoring
    rows = []
    for field in fields:
        if field not in regular_fields() or not (Path(out) / "splits" / f"{field}.json").exists():
            continue
        data = load_field(field); S = np.asarray(data["coordinates"], float)
        for d, r, t, tr in load_splits(out, field):
            if not d.startswith(("grid", "gsrs")):
                continue
            te = np.setdiff1d(np.arange(len(S)), tr)
            for target in R.tasks(field):
                y = data["Y"][:, F.C.TARGETS.index(target)]
                rows.append(sample_mean_rows(dict(field=field, design=d, rep=r, target=target, n=len(tr), m=len(te)), S, tr, te, y))
    if not rows:
        return
    s = scoring.score(pd.DataFrame(rows), {"DB": R.methods()[1]["DB"]})
    say("\n-- check 4: the sample mean's coverage of field means, grid against random samples of the same size --")
    for k in SPACINGS:
        a = R.coverage(s[s.design == f"grid{k}"], "DB", "field"); b = R.coverage(s[s.design == f"gsrs{k}"], "DB", "field")
        say(f"  spacing {k}: grid {a[0]:.3f} [{a[1][0]:.3f}, {a[1][1]:.3f}] ({a[2]} fields); random {b[0]:.3f} [{b[1][0]:.3f}, "
            f"{b[1][1]:.3f}] ({b[2]} fields) -> {'as expected' if a[0] >= b[0] else 'LOWER under grid: look for an error'}")


# ------------------------------------------------------------------------------------------------------------------------- TabPFN
def device_patch():
    """E21_DEVICE=cpu runs TabPFN on the CPU (the study's code fixes device='cuda'); the same settings otherwise."""
    dev = os.environ.get("E21_DEVICE")
    if not dev:
        return
    from experiments.core import common, point_model

    def tab(n_estimators=S_.N_ESTIMATORS, target_transform=False, seed=S_.RANDOM_STATE):
        common.single_thread()
        from tabpfn import TabPFNRegressor
        kw = {} if target_transform else {"inference_config": S_.INFERENCE_CONFIG}
        return TabPFNRegressor(device=dev, n_estimators=n_estimators, random_state=seed, **kw)
    point_model.tabpfn = tab


def point_outputs(root, field, data, design, rep, tr, target):
    from experiments.core.common import point_file, save_npz
    from experiments.core.point_model import our_model
    device_patch()
    path = point_file(root, "ours", design, field, target, rep)
    if path.exists():
        return False
    te = np.setdiff1d(np.arange(len(data["coordinates"])), tr); y = data["Y"][:, F.C.TARGETS.index(target)]
    arrays, info = our_model(field, data, design, rep, tr, te, target, "ours", ("Gbma",))
    save_npz(path, train_rows=tr, test_rows=te, observed=y[te], **arrays, info=json.dumps(info))
    return True


def stage_points(out, fields):
    from experiments.core.common import load_field
    root = Path(out) / "points"
    for field in fields:
        data = load_field(field)
        for d, r, t, tr in load_splits(out, field):
            for target in ((t,) if t else R.tasks(field)):
                t0 = time.monotonic()
                if point_outputs(root, field, data, d, r, tr, target):
                    print(f"{d} {field} {target} rep={r} n={len(tr)} {time.monotonic() - t0:.1f} s", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- rows
def stratified_row(key, S, tr, te, y):
    """The stratified estimator of the field's unrevealed mean: sum_h (m_h / m) ybar_{R_h}, variance sum_h (m_h / m)^2 s_h^2 (1/n_h + 1/m_h),
    t with n - 4 degrees of freedom (every zone needs two revealed samples)."""
    z = zone_labels(S); m = len(te); M = V = 0.0
    for h in range(S_.ZONE_K):
        r, u = tr[z[tr] == h], te[z[te] == h]
        if len(u) == 0:
            continue
        if len(r) < 2:
            return []
        w = len(u) / m; M += w * float(y[r].mean()); V += w ** 2 * float(y[r].var(ddof=1)) * (1 / len(r) + 1 / len(u))
    return [dict(key, block="field", mB=m, nB=len(tr), variant="DBs", M=M, V=V, df=float(len(tr) - S_.ZONE_K),
                 obs=float(y[te].mean()), failed=not (np.isfinite(M) and V > 1e-18), own_fallback=False, err="")]


def field_rows(job):
    out, root, field = job
    from experiments.core import block_cases
    from experiments.core.common import load_field
    from experiments.pipeline import further_models
    R.setup(R.CODES); block_cases.FINAL_POINTS = Path(root)
    shapes = json.loads((CONFIGS / "fixed" / "shapes.json").read_text())
    data = load_field(field); S = np.asarray(data["coordinates"], float); rows, missing, t0 = [], 0, time.monotonic()
    for d, r, t, tr in load_splits(out, field):
        for target in ((t,) if t else R.tasks(field)):
            case = block_cases.load_case(SET, field, data, d, r, tr, target)
            if case is None:
                missing += 1; continue
            rows += [x for x in block_cases.case_rows(case, shapes, ours=True) if not x["variant"].startswith("KpR")]
            rows += [x for x in further_models.case_rows(case)[0] if x["variant"] in ("O1", "DE", "DEz")]
            if d.startswith("strat"):
                rows += stratified_row(dict(set=SET, src="external", field=field, design=d, rep=r, target=target, n=len(tr),
                                            m=len(case["te"])), S, tr, case["te"], case["y"])
    p = Path(out) / "rows"; p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p / f"rows_{SET}_{field}.csv", index=False)
    return field, len(rows), missing, time.monotonic() - t0


def stage_rows(out, fields, workers):
    root = Path(out) / "points"
    with Pool(min(workers, len(fields))) as pool:
        for field, n, missing, sec in pool.imap_unordered(field_rows, [(out, str(root), f) for f in fields]):
            print(f"rows {field}: {n} rows, {missing} cases without point outputs, {sec:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- check 1
def stage_check(out, study_rows):
    """Check 1: the study's own splits (n = 20, repeats 300 and 301) recomputed through this module reproduce the stored rows."""
    from experiments.core import block_cases
    from experiments.core.common import load_field, training_rows
    from experiments.pipeline import further_models
    R.setup(R.CODES); root = Path(out) / "check" / "points"; block_cases.FINAL_POINTS = root
    shapes = json.loads((CONFIGS / "fixed" / "shapes.json").read_text())
    rows = []
    for field in CHECK_FIELDS:
        data = load_field(field)
        for rep in (300, 301):
            tr = training_rows(data, field, "n20", rep)
            for target in R.tasks(field):
                point_outputs(root, field, data, "n20", rep, tr, target)
                case = block_cases.load_case("check", field, data, "n20", rep, tr, target)
                rows += block_cases.case_rows(case, shapes, ours=True)
                rows += [x for x in further_models.case_rows(case)[0] if x["variant"] in ("O1", "DE", "DEz")]
    new = pd.DataFrame(rows)
    old = stored_rows(study_rows, CHECK_FIELDS)
    old = old[(old.design == "n20") & old.rep.isin([300, 301])]
    say("=== E21 check 1: the study's splits through this module against the stored rows ===")
    worst = 0.0
    for v in ("J1p", "O1", "TAB", "RK_ins_fix_v", "WH_fix", "DB"):
        a = new[new.variant == v].set_index(R.KEYS)[["M", "V"]]; b = old[old.variant == v].set_index(R.KEYS)[["M", "V"]]
        j = a.join(b, lsuffix="_new", rsuffix="_old", how="inner")
        rel = max(float((np.abs(j.M_new - j.M_old) / np.maximum(np.abs(j.M_old), 1e-12)).max()),
                  float((np.abs(j.V_new - j.V_old) / np.maximum(np.abs(j.V_old), 1e-12)).max()))
        worst = max(worst, rel)
        say(f"  {v:13s} {len(j)} of {len(b)} blocks matched; largest relative difference in M or V {rel:.2e}")
    say(f"check 1 {'passed' if worst < 1e-6 else 'FAILED: stop'} (largest {worst:.2e})")
    (Path(out) / "check1.txt").write_text("\n".join(lines) + "\n")


def stored_rows(study_rows, fields=None):
    p = Path(study_rows); parts = []
    for f in sorted(p.glob("rows_study_*.csv")):
        parts.append(pd.read_csv(f, low_memory=False))
    for pat in ("models_study_*.csv", "revision_study_*.csv"):
        for f in sorted(p.glob(pat)):
            q = pd.read_csv(f, low_memory=False); parts.append(q[q.variant.isin(["O1", "DE", "DEz"])])
    r = pd.concat(parts, ignore_index=True); r["failed"] = r.failed.astype(bool)
    return r[r.field.isin(fields)] if fields else r


# ------------------------------------------------------------------------------------------------------------------------- report
NAMES = {"J1p": "TabPFN-GP", "O1": "Shared-error-only", "TAB": "TabPFN", "RK_ins_fix_v": "Block RK", "WH_fix": "WH", "DB": "Sample mean",
         "DE": "Difference estimator", "DEz": "Difference estimator", "DBs": "Stratified estimator"}
csv_rows = []


def stage_report(out, study_rows):
    from experiments.core import scoring
    _, factors, crps_f, _ = R.methods()
    factors = {**factors, "DE": FACTOR_DE, "DEz": FACTOR_DE, "DBs": factors["DB"]}
    crps_f = {**crps_f, "DE": 1.0, "DEz": 1.0, "DBs": 1.0}
    new = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted((Path(out) / "rows").glob(f"rows_{SET}_*.csv"))], ignore_index=True)
    ref = stored_rows(study_rows); ref = ref[ref.design.isin([f"n{n}" for n in SIZES])]
    raw = pd.concat([new, ref], ignore_index=True); raw["failed"] = raw.failed.astype(bool)
    raw = raw[raw.variant.isin(list(NAMES))]
    sc = scoring.score(raw, factors); sc["crps"] = scoring.score(raw, crps_f)["crps"].to_numpy()
    sc["dfam"] = np.where(sc.design.str.match(r"n\d+$"), "srs", sc.design.str.replace(r"\d+$", "", regex=True))
    say("=== E21: other sampling designs on the 43 fields (docs/exploratory/plan_E21.md; exploratory) ===")
    say("coverage of 80 % intervals with each method's study factor; [95 % interval over fields] (fields); change from the reference, paired by field")
    comps = ["TAB", "RK_ins_fix_v", "WH_fix"]
    groups = [(f"{d}{n}", "srs", n) for d in FIXED for n in SIZES] + [(d, "srs", None) for d in FIXED] + \
             [(f"grid{k}", f"gsrs{k}", None) for k in SPACINGS] + [("grid", "gsrs", None)]
    for g, refname, n in groups:
        A = select(sc, g); B = select(sc, refname, n=n, like=A)
        if A.empty:
            continue
        say(f"\n-- {g} against {refname if refname != 'srs' else 'the study'}" + (f" (n = {n})" if n else "") + " --")
        for kind in ("field", "zone", "nozone"):
            for v in ["J1p", "O1"] + comps + ["DB", "DE" if kind == "field" else "DEz"] + (["DBs"] if g.startswith("strat") and kind == "field" else []):
                a = cov_set(A, v, comps); b = cov_set(B, v, comps)
                ka = "zone" if kind == "nozone" else kind
                if kind == "nozone":
                    a, b = a[(a.block != "field") & (a.nB == 0)], b[(b.block != "field") & (b.nB == 0)]
                e, ci, nf = R.coverage(a, v, ka)
                if not np.isfinite(e):
                    continue
                d, dci = paired_change(a, b, v, ka)
                say(f"  {kind:6s} {NAMES[v]:22s} {e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] ({nf}); change {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")
                csv_rows.append(dict(design=g, reference=refname, n=n or "", kind=kind, method=NAMES[v], variant=v, coverage=e, lo=ci[0],
                                     hi=ci[1], fields=nf, change=d, change_lo=dci[0], change_hi=dci[1]))
        for kind in ("field", "zone"):
            for v in ["O1"] + comps + ["DB", "DE" if kind == "field" else "DEz"]:
                for col in ("crps", "width"):
                    c = scoring.contrast(pair_set(A, v, comps), "J1p", v, kind, col=col)
                    if c:
                        say(f"  {kind:6s} {col} ratio TabPFN-GP / {NAMES[v]:22s} {c['ratio']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] "
                            f"({c['better']}/{c['fields']} fields lower)")
                        csv_rows.append(dict(design=g, reference=refname, n=n or "", kind=kind, method=NAMES[v], variant=v,
                                             **{f"{col}_ratio": c["ratio"], f"{col}_lo": c["ci"][0], f"{col}_hi": c["ci"][1]}))
        z = A[(A.block != "field")]
        for lab, q in (("none", z[z.nB == 0]), ("one or two", z[z.nB.isin([1, 2])]), ("three or more", z[z.nB >= 3])):
            e, ci, nf = R.coverage(cov_set(q, "J1p", comps), "J1p", "zone")
            if np.isfinite(e):
                say(f"  zones with {lab} revealed samples: TabPFN-GP {e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] ({nf} fields)")
    (Path(out) / "report11.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(Path(out) / "report11.csv", index=False)


def select(sc, name, n=None, like=None):
    """Rows of a design family or design; for the reference of the fixed-n designs, the study's rows at the same n (all three if
    n is None), on the fields of `like`."""
    if name == "srs":
        s = sc[sc.dfam == "srs"]
        s = s[s.n == n] if n else s[s.n.isin(SIZES)]
    elif name in ("grid", "gsrs"):
        s = sc[sc.dfam == name]
    elif name in FIXED:
        s = sc[sc.dfam == name]
    else:
        s = sc[sc.design == name]
    if like is not None:
        s = s[s.field.isin(like.field.unique())]
        if name == "srs":
            s = s[s.target.isin(like.target.unique())]
    return s


def cov_set(s, v, comps):
    """The blocks a method's coverage is computed on, as in the study: TabPFN-GP and the three comparators on the blocks where all four
    are scored; the other methods on their own."""
    if v in ["J1p"] + comps:
        from experiments.core import scoring
        return scoring.common_blocks(s, ["J1p"] + comps)
    return s


def pair_set(s, v, comps):
    from experiments.core import scoring
    return scoring.common_blocks(s, ["J1p", v])


def paired_change(a, b, v, kind):
    """Coverage of a minus coverage of b over the fields with both, each pooled over blocks; 95 % interval from resampling fields."""
    def per_field(s):
        q = s[(s.variant == v) & (R.kind_of(s) == kind) & s.cov80.notna()]
        return q.groupby("field").cov80.agg(["sum", "count"])
    ga, gb = per_field(a), per_field(b); f = ga.index.intersection(gb.index)
    if len(f) == 0:
        return np.nan, (np.nan, np.nan)
    ga, gb = ga.loc[f], gb.loc[f]; idx = R.boot_idx(len(f))
    est = ga["sum"].sum() / ga["count"].sum() - gb["sum"].sum() / gb["count"].sum()
    bs = ga["sum"].to_numpy()[idx].sum(1) / ga["count"].to_numpy()[idx].sum(1) - gb["sum"].to_numpy()[idx].sum(1) / gb["count"].to_numpy()[idx].sum(1)
    return float(est), tuple(np.percentile(bs, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["splits", "check", "points", "rows", "report"])
    ap.add_argument("--out", required=True); ap.add_argument("--fields", nargs="*"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--study-rows")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    R.setup(R.CODES); fields = list(a.fields or R.evaluated())
    if a.stage == "splits":
        stage_splits(a.out, fields)
    elif a.stage == "check":
        stage_check(a.out, a.study_rows)
    elif a.stage == "points":
        stage_points(a.out, fields)
    elif a.stage == "rows":
        stage_rows(a.out, fields, a.workers)
    else:
        stage_report(a.out, a.study_rows)


if __name__ == "__main__":
    main()
