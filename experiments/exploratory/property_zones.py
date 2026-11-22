"""E23 of docs/exploratory/plan_E23.md: zones formed from the soil property on the 43 fields (exploratory).

For each field and property, each location gets the mean of the property at its four nearest other locations (its own value left out),
and these neighbour means are clustered by k-means into four zones (random seed 0, best of ten starts), at all the field's locations.
The study's splits and stored point outputs are used; only the zones change. The functions of E22 (covariate_zones.py) compute every
method with the study's own code, here with the zones keyed by field and property. No existing module is changed.

  python -m experiments.exploratory.property_zones zones --out DIR                   (CPU)    zones, checks 2 and 4 -> DIR/zones/, DIR/check2.txt
  python -m experiments.exploratory.property_zones check --out DIR --study-rows ST --variants VA   (CPU, R) check 1 -> DIR/check1.txt
  python -m experiments.exploratory.property_zones rows --out DIR [--workers 12]     (CPU, R) DIR/rows/rows_e23_<field>_<design>.csv
  python -m experiments.exploratory.property_zones report --out DIR --study-rows ST --variants VA   -> DIR/report13.txt, DIR/report13.csv
Run from the repository root with SOILGP_DATA and SOILGP_RUN set to the study's data and run folders; ST and VA as for covariate_zones.py.
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
from scipy.spatial import cKDTree                                             # noqa: E402
from sklearn.cluster import KMeans                                            # noqa: E402
from sklearn.metrics import adjusted_rand_score                               # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.exploratory import covariate_zones as CZ                     # noqa: E402
from experiments.study import run as R                                        # noqa: E402
from experiments.paths import CONFIGS                                         # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402

SET = "e23"
CZ.SET = SET                                                                  # the set name of the rows CZ.case_rows writes
NEIGHBOURS = 4
METHODS, NAMES, COMPS, PUBLIC = CZ.METHODS, CZ.NAMES, CZ.COMPS, CZ.PUBLIC
LABELS = {}                                                                   # (field, property) -> zone label of every location
say, lines = CZ.say, CZ.lines


# ------------------------------------------------------------------------------------------------------------------------- zones
def neighbours(S):
    """The indices of each location's NEIGHBOURS nearest other locations (no two locations share coordinates)."""
    d, idx = cKDTree(S).query(S, k=NEIGHBOURS + 1)
    assert np.all(idx[:, 0] == np.arange(len(S))) and np.all(d[:, 1] > 0), "duplicate coordinates"
    return idx[:, 1:]


def property_labels(S, y):
    """Zones from the property: k-means of the mean of y at each location's nearest other locations."""
    assert np.all(np.isfinite(y)), "missing property values"
    sm = y[neighbours(S)].mean(1)
    return KMeans(S_.ZONE_K, random_state=0, n_init=10).fit_predict(sm.reshape(-1, 1))


def contiguity(S, lab):
    """Share of locations whose nearest other location lies in the same zone."""
    return float(np.mean(lab[neighbours(S)[:, 0]] == lab))


def blocks_of(case):
    """block_cases.blocks_of with the zones of LABELS for the case's field and property."""
    lab = LABELS[(case["field"], case["target"])]; tr, te = case["tr"], case["te"]
    blocks = {"field": np.arange(len(te))}
    for j in range(int(lab.max()) + 1):
        idx = np.flatnonzero(lab[te] == j)
        if len(idx) >= S_.ZONE_MIN_POINTS:
            blocks[f"zone{j}"] = idx
    revealed = {b: (tr if b == "field" else tr[lab[tr] == int(b[4:])]) for b in blocks}
    return blocks, revealed


def install(field, labels):
    """Use the given zones ({property: labels}) in the study's code (this process only); skip the local-variogram R part, as in E22."""
    from experiments.core import block_cases
    for t, lab in labels.items():
        LABELS[(field, t)] = np.asarray(lab, int)
    if block_cases.blocks_of is not blocks_of:
        block_cases.blocks_of = blocks_of
        run_r = block_cases.run_r

        def fixed_only(mode, S, tr, te, columns, params=None):
            if params is not None and params.get("parts") == ["loc"]:
                return {}, {}
            return run_r(mode, S, tr, te, columns, params)
        block_cases.run_r = fixed_only
    return block_cases


def load_zones(out, field):
    d = json.loads((Path(out) / "zones" / f"{field}.json").read_text())
    return np.asarray(d["compact"], int), {t: np.asarray(v, int) for t, v in d["property"].items()}


def stage_zones(out, fields):
    """The zones of every field and property; checks 2 and 4."""
    from experiments.core.common import load_field
    say("=== E23 zones and check 2 (docs/exploratory/plan_E23.md) ===")
    say("field, property: property-zone sizes (all locations); adjusted Rand index with the compact | covariate zones; share of the "
        "property's variance between zones, property | covariate | compact; contiguity (nearest other location in the same zone), "
        "property | covariate | compact")
    desc = []
    for f in fields:
        data = load_field(f); S = np.asarray(data["coordinates"], float)
        comp, cov = CZ.labels_of(data, "compact"), CZ.labels_of(data, "covariate"); prop = {}
        for t in R.tasks(f):
            y = np.asarray(data["Y"][:, F.C.TARGETS.index(t)], float); lab = property_labels(S, y); prop[t] = lab
            sizes = np.bincount(lab, minlength=S_.ZONE_K)
            r = dict(field=f, target=t, sizes="/".join(map(str, sizes)), min_size=int(sizes.min()),
                     ari_compact=adjusted_rand_score(comp, lab), ari_covariate=adjusted_rand_score(cov, lab),
                     share_property=CZ.between_share(y, lab), share_covariate=CZ.between_share(y, cov), share_compact=CZ.between_share(y, comp),
                     contig_property=contiguity(S, lab), contig_covariate=contiguity(S, cov), contig_compact=contiguity(S, comp))
            small = [int(j) for j in range(S_.ZONE_K) if sizes[j] < S_.ZONE_MIN_POINTS]
            say(f"  {f} {t.replace('_target', '')}: sizes {r['sizes']}; ARI {r['ari_compact']:.2f} | {r['ari_covariate']:.2f}; share "
                f"{r['share_property']:.2f} | {r['share_covariate']:.2f} | {r['share_compact']:.2f}; contiguity {r['contig_property']:.2f} | "
                f"{r['contig_covariate']:.2f} | {r['contig_compact']:.2f}"
                + (f"; zone(s) {small} never evaluated (fewer than {S_.ZONE_MIN_POINTS} locations)" if small else ""))
            desc.append(r)
        p = Path(out) / "zones" / f"{f}.json"; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"compact": comp.tolist(), "property": {t: v.tolist() for t, v in prop.items()}}))
    d = pd.DataFrame(desc); d.to_csv(Path(out) / "zones" / "zones.csv", index=False)
    med = lambda c: f"{d[c].median():.3f} ({d[c].min():.2f}-{d[c].max():.2f})"          # noqa: E731
    say(f"\n{d.field.nunique()} fields, {len(d)} field-property pairs. Medians (range): ARI with the compact zones {med('ari_compact')}, "
        f"with the covariate zones {med('ari_covariate')}; share between zones: property {med('share_property')}, covariate "
        f"{med('share_covariate')}, compact {med('share_compact')} (property higher than compact on "
        f"{int((d.share_property > d.share_compact).sum())} of {len(d)} pairs); contiguity: property {med('contig_property')}, covariate "
        f"{med('contig_covariate')}, compact {med('contig_compact')}; smallest property zone {d.min_size.median():.0f} locations "
        f"({d.min_size.min()}-{d.min_size.max()}); pairs with a zone never evaluated {int((d.min_size < S_.ZONE_MIN_POINTS).sum())}")
    ok = d.share_property.median() > d.share_compact.median()
    say(f"check 4 {'passed' if ok else 'FAILED: the property zones are not sharper than the compact zones; report check 2 only'} "
        f"(median share between zones {d.share_property.median():.3f} against {d.share_compact.median():.3f})")
    (Path(out) / "check2.txt").write_text("\n".join(lines) + "\n")
    if not ok:
        sys.exit(2)


# ------------------------------------------------------------------------------------------------------------------------- rows
def job_rows(job):
    """Rows of one field and design (all repeats and properties) with the property zones."""
    out, field, design = job
    from experiments.core.common import load_field
    R.setup(R.CODES); shapes = json.loads((CONFIGS / "fixed" / "shapes.json").read_text())
    _, prop = load_zones(out, field); block_cases = install(field, prop); zone_block = CZ.zone_effect()
    data = load_field(field); rows, missing, t0 = [], 0, time.monotonic()
    for f, data, d, rep, tr in block_cases.iter_splits("study", [field]):
        if d != design:
            continue
        for target in R.tasks(field):
            case = block_cases.load_case("study", field, data, d, rep, tr, target)
            if case is None:
                missing += 1; continue
            rows += CZ.case_rows(case, prop[target], shapes, zone_block)
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
def stage_check(out, study_rows, variants):
    """Check 1: with the compact zones installed for every property, this module reproduces the stored rows of every method."""
    from experiments.core.common import load_field, training_rows
    R.setup(R.CODES); shapes = json.loads((CONFIGS / "fixed" / "shapes.json").read_text()); zone_block = CZ.zone_effect()
    rows = []
    for field in CZ.CHECK_FIELDS:
        data = load_field(field); comp = CZ.labels_of(data, "compact"); block_cases = install(field, {t: comp for t in R.tasks(field)})
        for rep in (300, 301):
            tr = training_rows(data, field, "n20", rep)
            for target in R.tasks(field):
                case = block_cases.load_case("study", field, data, "n20", rep, tr, target)
                rows += CZ.case_rows(case, comp, shapes, zone_block)
    new = pd.DataFrame(rows)
    old = CZ.stored_rows(study_rows, variants, list(CZ.CHECK_FIELDS)); old = old[(old.design == "n20") & old.rep.isin([300, 301])]
    say("=== E23 check 1: the compact zones through this module against the stored rows ===")
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
    if worst >= 1e-6:
        sys.exit(2)


# ------------------------------------------------------------------------------------------------------------------------- report
csv_rows = []


def stage_report(out, study_rows, variants):
    from experiments.core import scoring
    R.setup(R.CODES)
    fmt = lambda e, ci: f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"                  # noqa: E731
    new = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted((Path(out) / "rows").glob(f"rows_{SET}_*.csv"))], ignore_index=True)
    new["failed"] = new.failed.astype(bool)
    fields = sorted(new.field.unique()); old = CZ.stored_rows(study_rows, variants, fields)
    sc_new, factors, crps_f = CZ.scored(new); sc_old, _, _ = CZ.scored(old)
    sets = {"property": sc_new, "compact": sc_old}
    ms = CZ.method_set
    for name in ("check1.txt", "check2.txt"):                                # checks 1, 2 and 4, as written by their stages
        p = Path(out) / name
        if p.exists():
            say(p.read_text().rstrip() + "\n")
    say("=== E23: zones formed from the soil property on the 43 fields (docs/exploratory/plan_E23.md; exploratory) ===")
    say("80 % intervals with each method's study factor (" + ", ".join(f"{NAMES[v]} {factors[v]:.2f}" for v in METHODS) + "); "
        "[95 % interval over fields] (fields); change from the compact zones on the same splits, paired by field")

    a = new[(new.block == "field") & (new.variant != "Z")].set_index(R.KEYS + ["variant"])[["M", "V"]]
    b = old[(old.block == "field") & (old.variant != "Z")].set_index(R.KEYS + ["variant"])[["M", "V"]]
    j = a.join(b, lsuffix="_new", rsuffix="_old", how="inner")
    rel = max(float((np.abs(j.M_new - j.M_old) / np.maximum(np.abs(j.M_old), 1e-12)).max()),
              float((np.abs(j.V_new - j.V_old) / np.maximum(np.abs(j.V_old), 1e-12)).max()))
    say(f"field rows against the stored ones: {len(j)} of {len(b)} matched ({len(a)} new); largest relative difference in M or V {rel:.2e}")
    say("zone means scored: " + "; ".join(f"{k}: " + ", ".join(f"{NAMES[v]} {len(ms(s, v))}" for v in METHODS) for k, s in sets.items()))
    say("failed zone predictions (property zones): " + ", ".join(
        f"{NAMES[v]} {int(new[(new.variant == v) & (new.block != 'field')].failed.sum())}/{int(((new.variant == v) & (new.block != 'field')).sum())}"
        for v in METHODS))

    say("\n-- check 3: the sample mean's coverage of property-zone means (theory: near 0.80 under simple random sampling) --")
    e, ci, k = R.coverage(ms(sc_new, "DB"), "DB", "zone"); e0, ci0, _ = R.coverage(ms(sc_old, "DB"), "DB", "zone")
    ok = e >= R.BAND[0]
    say(f"  property zones {fmt(e, ci)} ({k} fields); compact zones {fmt(e0, ci0)} -> {'passed' if ok else 'FAILED: look for an error'}")
    if not ok:
        (Path(out) / "report13.txt").write_text("\n".join(lines) + "\n"); sys.exit(2)

    def compare(section, subset, A, B, v, indent="  "):
        e, ci, k = R.coverage(A, v, "zone")
        if not np.isfinite(e):
            return
        e0, ci0, _ = R.coverage(B, v, "zone"); d, dci = CZ.paired_change(A, B)
        tail = f" -> {CZ.verdict(e, ci)}" if section == "coverage" else ""
        say(f"{indent}{NAMES[v]:22s} {fmt(e, ci)} ({k}; {len(A)} means) | {fmt(e0, ci0)} | {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]{tail}")
        csv_rows.append(dict(section=section, method=NAMES[v], variant=v, subset=subset, coverage=e, lo=ci[0], hi=ci[1], fields=k,
                             compact=e0, compact_lo=ci0[0], compact_hi=ci0[1], change=d, change_lo=dci[0], change_hi=dci[1],
                             reading=CZ.verdict(e, ci) if section == "coverage" else ""))

    say("\n-- coverage of zone means: property zones | compact zones | change; reading (band [0.75, 0.85]) --")
    for v in METHODS:
        compare("coverage", "all", ms(sc_new, v), ms(sc_old, v), v)

    say("\n-- by the zone's number of revealed samples: property | compact | change --")
    for lab, sel in (("none", lambda q: q.nB == 0), ("one or two", lambda q: q.nB.isin([1, 2])), ("three or more", lambda q: q.nB >= 3)):
        say(f"  zones with {lab} revealed samples")
        for v in METHODS:
            A, B = ms(sc_new, v), ms(sc_old, v)
            compare("revealed", lab, A[sel(A)], B[sel(B)], v, "    ")

    zd = pd.read_csv(Path(out) / "zones" / "zones.csv")
    cut = np.quantile(zd.share_property, [1 / 3, 2 / 3])
    zd["third"] = np.where(zd.share_property <= cut[0], "least sharp third", np.where(zd.share_property <= cut[1], "middle third", "sharpest third"))
    say(f"\n-- by property, by how sharply the zones separate the property (thirds of the field-property pairs by the property zones' "
        f"share between zones, cut at {cut[0]:.3f} and {cut[1]:.3f}), by group of fields, by covariate source: property | compact | change --")
    pairs = lambda sub: set(zip(sub.field, sub.target))                       # noqa: E731
    subsets = [(t.replace("_target", ""), pairs(zd[zd.target == t])) for t in sorted(zd.target.unique())]
    subsets += [(th, pairs(zd[zd.third == th])) for th in ("least sharp third", "middle third", "sharpest third")]
    subsets += [(g, pairs(zd[zd.field.map(R.group) == g])) for g in ("Uniformity trials", "SensRes", "Midwest", "New fields", "Further fields")]
    subsets += [("released covariates", pairs(zd[~zd.field.isin(PUBLIC)])), ("public covariates", pairs(zd[zd.field.isin(PUBLIC)]))]
    for lab, pr in subsets:
        say(f"  {lab} ({len(pr)} field-property pairs, {len({f for f, _ in pr})} fields)")
        for v in METHODS:
            A, B = ms(sc_new, v), ms(sc_old, v)
            inA = [p in pr for p in zip(A.field, A.target)]; inB = [p in pr for p in zip(B.field, B.target)]
            compare("subset", lab, A[inA], B[inB], v, "    ")
    ns = sorted(new.n.unique())
    for v in METHODS:
        parts = []
        for n in ns:
            A, B = ms(sc_new, v), ms(sc_old, v); A, B = A[A.n == n], B[B.n == n]
            e, ci, k = R.coverage(A, v, "zone")
            if np.isfinite(e):
                e0 = R.coverage(B, v, "zone")[0]; parts.append(f"{n}: {e:.3f} | {e0:.3f}")
                csv_rows.append(dict(section="by_n", method=NAMES[v], variant=v, subset=f"n={n}", coverage=e, lo=ci[0], hi=ci[1], fields=k, compact=e0))
        say(f"  {NAMES[v]:22s} by n " + "; ".join(parts))

    say("\n-- rise in coverage D of zone means (n >= 30 minus n <= 10): property zones [90 %] [95 %] | compact zones --")
    for v in METHODS:
        out_ = []
        for k_, s in sets.items():
            q = ms(s, v); est, bs, nf = R.rise(q, v, "zone")
            out_.append((est, np.percentile(bs, [5, 95]), np.percentile(bs, [2.5, 97.5]), nf))
        (e, c90, c95, nf), (e0, _, c950, _) = out_
        say(f"  {NAMES[v]:22s} D = {e:+.3f} [{c90[0]:+.3f}, {c90[1]:+.3f}] [{c95[0]:+.3f}, {c95[1]:+.3f}] ({nf}) | {e0:+.3f} "
            f"[{c950[0]:+.3f}, {c950[1]:+.3f}]")
        csv_rows.append(dict(section="rise", method=NAMES[v], variant=v, subset="D", coverage=e, lo=c95[0], hi=c95[1], lo90=c90[0], hi90=c90[1],
                             fields=nf, compact=e0, compact_lo=c950[0], compact_hi=c950[1]))

    say("\n-- CRPS and interval-width ratio TabPFN-GP / method for zone means: property zones | compact zones --")
    for v in METHODS[1:]:
        for col in ("crps", "width"):
            res = []
            for k_, s in sets.items():                                    # the study's sets, as in E22
                common = scoring.common_blocks(s, ["J1p"] + COMPS)
                q = (common if v in COMPS else s if v in ("O1", "Z")
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

    say("\n-- coverage by field, TabPFN-GP and the sample mean: property | compact --")
    for f in fields:
        a1, b1 = ms(sc_new, "J1p"), ms(sc_old, "J1p"); a2, b2 = ms(sc_new, "DB"), ms(sc_old, "DB")
        say(f"  {f:8s} TabPFN-GP {R.coverage(a1, 'J1p', 'zone', [f])[0]:.3f} | {R.coverage(b1, 'J1p', 'zone', [f])[0]:.3f}; "
            f"sample mean {R.coverage(a2, 'DB', 'zone', [f])[0]:.3f} | {R.coverage(b2, 'DB', 'zone', [f])[0]:.3f}; "
            f"zone means {len(a1[a1.field == f])} | {len(b1[b1.field == f])}")
    (Path(out) / "report13.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(Path(out) / "report13.csv", index=False)


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
