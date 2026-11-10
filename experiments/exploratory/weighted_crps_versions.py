"""The addition to E19 in docs/exploratory/plan_E19.md (CPU): the weighted CRPS (1) with the study's versions, factors and CRPS versions and
(2) with each comparator's version that has the lowest weighted CRPS on the development splits.

  SOILGP_DATA=<work>/data SOILGP_RUN=<work>/run python -m experiments.exploratory.weighted_crps_versions <factor-setting folder> <work>/rows <work>/report/report.txt <E19 folder> <out>
<E19 folder> is the output folder of weighted_crps.py. Writes <out>/unit (the check), <out>/kept (1), <out>/lowest (2: versions, factors,
CRPS versions and the study's report), <out>/report9b.txt and report9b.csv.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A, ROWS, STORED, E19, OUT = (Path(a) for a in sys.argv[1:6])
from experiments.study import run as S                                        # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.factor_setting import set_factors                            # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
CONFIGS = S.CONFIGS
ORIGINAL = scoring.score
KEYS = ("variant", "factor", "crps_version")
H5 = re.compile(r"^H5 (field|zone)\s+TabPFN-GP / (.+?)\s+CRPS ratio ([\d.]+) \[([\d.]+), ([\d.]+)\], lower on (\d+) of (\d+) fields, "
                r"Wilcoxon p = (\S+), Holm (\S+): (.+)$")
H4 = re.compile(r"^H4\(a\) .*CRPS ratio to TabPFN-GP ([\d.]+) \[([\d.]+), ([\d.]+)\] \((\d+) fields\): (.+)$")
H1 = re.compile(r"^H1 (.+?)\s+field-mean D = ([+-][\d.]+), 95 % \[([+-][\d.]+), ([+-][\d.]+)\] \((\d+) fields\)$")
H1_ALL = re.compile(r"^H1 \(.*\): (.+)$")
TAGS = ("unweighted", "weights alone", "weights, lowest versions")
lines, csv_rows = [], []


def say(s=""):
    lines.append(s); print(s, flush=True)


def weighted(rows, factors=None):
    r = ORIGINAL(rows, factors)
    r["crps"] = r["crps"].to_numpy() * (r.mB.to_numpy(float) / (r.mB.to_numpy(float) + r.nB.to_numpy(float)))
    return r


def report(score, configs, folder):
    scoring.score = score
    S.CONFIGS = configs
    S.stage_report(str(ROWS), str(folder), S.evaluated())
    return folder / "report.txt"


def tests(path):
    t = path.read_text().splitlines()
    return ([m.groups() for m in map(H5.match, t) if m], next(m.groups() for m in map(H4.match, t) if m),
            [m.groups() for m in map(H1.match, t) if m], next(m.group(1) for m in map(H1_ALL.match, t) if m))


def comparators(folder):
    return json.loads((folder / "calibration.json").read_text())["comparators"]


def no_crps(path):
    return [x for x in path.read_text().splitlines() if "crps" not in x.lower()]


def same_h5(a, b, groups, what):
    for x, y in zip(a, b):
        assert x[:2] == y[:2], (x[:2], y[:2])
        if x[1] in groups:
            assert x[2] == y[2] and x[5] == y[5], (what, x, y)


# ------------------------------------------------------------------------------------------------------------------- versions
w_fixed = E19 / "weighted" / "fixed"
scores = pd.read_csv(w_fixed / "selection_scores.csv")
sel_w = json.loads((w_fixed / "selection.json").read_text())
sel_low = dict(sel_w)
for g, d in scores.groupby("group", sort=False):
    sel_low[g] = d.sort_values("score").variant.iloc[0]
    assert d[d.selected].variant.tolist() == [sel_w[g]], (g, "selection_scores.csv and selection.json disagree")
low = OUT / "lowest"; low_fixed = low / "fixed"; low_fixed.mkdir(parents=True, exist_ok=True)
(low_fixed / "selection.json").write_text(json.dumps(sel_low, indent=1) + "\n")
scoring.score = weighted
set_factors.calibrate(str(A), str(low_fixed / "selection.json"), str(low_fixed))
cal_study, cal_w, cal_low = comparators(CONFIGS / "fixed"), comparators(w_fixed), comparators(low_fixed)

# ------------------------------------------------------------------------------------------------------------------- checks
unit = report(ORIGINAL, CONFIGS, OUT / "unit")
assert unit.read_bytes() == STORED.read_bytes(), "check 1: report differs from the study's"
print("check 1 passed: with weights of 1 the report is identical to the study's", flush=True)

kept = report(weighted, CONFIGS, OUT / "kept")
assert no_crps(kept) == no_crps(STORED), "check 2: a line without the CRPS changed"
print("check 2 passed: every line without the CRPS is identical to the study's", flush=True)

lowest = report(weighted, low, low / "report")
h5_u, h4_u, h1_u, t1_u = tests(STORED)
h5_k, h4_k, h1_k, t1_k = tests(kept)
h5_l, h4_l, h1_l, t1_l = tests(lowest)
h5_w, h4_w, _, _ = tests(E19 / "weighted" / "report" / "report.txt")

alike = [g for g in cal_study if all(cal_study[g][k] == cal_w[g][k] for k in KEYS)]
same_h5(h5_k, h5_w, alike, "check 3")
assert h4_k[0] == h4_w[0], ("check 3: test 4a", h4_k, h4_w)
print(f"check 3 passed: {', '.join(alike)} and test 4a as in the steps above", flush=True)

alike_low = [g for g in sel_w if sel_low[g] == sel_w[g]]
for g in alike_low:
    assert all(cal_low[g][k] == cal_w[g][k] for k in KEYS), ("check 4", g, cal_low[g], cal_w[g])
same_h5(h5_l, h5_w, alike_low, "check 4")
assert h4_l[0] == h4_w[0], ("check 4: test 4a", h4_l, h4_w)
print(f"check 4 passed: {', '.join(alike_low)} and test 4a as in the steps above", flush=True)

# ------------------------------------------------------------------------------------------------------------------- summary
say("=== E19, addition (docs/exploratory/plan_E19.md; specified after the study's results had been seen; exploratory) ===")
say("CRPS of each mean multiplied by m_B / (m_B + n_B); TabPFN-GP with kappa = 1.15.")
say("  weights alone:            the study's versions, factors and CRPS versions (configs/fixed/)")
say("  weights, lowest versions: each comparator's version with the lowest weighted CRPS on the development splits; factor and CRPS version")
say("                            set on the factor-setting data with the weighted CRPS")
say("\n-- comparators: version, factor, CRPS version; weighted CRPS on the development splits relative to TabPFN's (lower is better) --")
score_of = {(r.group, r.variant): r.score for r in scores.itertuples()}
for g in cal_study:
    for tag, c in zip(TAGS[1:], (cal_study[g], cal_low[g])):
        s = score_of.get((g, c["variant"]))
        say(f"  {g if tag == TAGS[1] else '':12s} {tag:25s} {c['variant']}, factor {c['factor']:.2f}, CRPS {c['crps_version']}"
            + ("" if s is None else f", score {s:.4f}"))
    a, b = score_of.get((g, cal_study[g]["variant"])), score_of.get((g, cal_low[g]["variant"]))
    if a is not None and b is not None and cal_low[g]["variant"] != cal_study[g]["variant"]:
        say(f"  {'':12s} {'':25s} score ratio {b / a:.4f}")

say("\n-- test 5: CRPS ratio TabPFN-GP / method [95 % interval], fields lower, Holm p, verdict --")
differ = {t: [] for t in TAGS[1:]}
above = {t: [] for t in TAGS[1:]}
for u, k, l in zip(h5_u, h5_k, h5_l):
    assert u[:2] == k[:2] == l[:2], (u[:2], k[:2], l[:2])
    for tag, x, head in zip(TAGS, (u, k, l), (f"{u[0]:5s} {u[1]:12s}", " " * 18, " " * 18)):
        say(f"  {head} {tag:25s} {float(x[2]):.3f} [{float(x[3]):.3f}, {float(x[4]):.3f}], {x[5]}/{x[6]}, Holm {float(x[8]):.3g}: {x[9]}")
        csv_rows.append(dict(test="5", kind=x[0], method=x[1], weighting=tag, ratio=float(x[2]), lo=float(x[3]), hi=float(x[4]), lower=int(x[5]),
                             fields=int(x[6]), holm=float(x[8]), verdict=x[9]))
        if tag != TAGS[0]:
            if x[9] != u[9]:
                differ[tag].append(f"{u[0]} / {u[1]}: {u[9]} -> {x[9]}")
            if float(x[2]) > 1:
                above[tag].append(f"{x[0]} / {x[1]} ({float(x[2]):.3f})")

say("\n-- test 4a: CRPS ratio shared-error-only / TabPFN-GP --")
for tag, x in zip(TAGS, (h4_u, h4_k, h4_l)):
    say(f"  {tag:25s} {float(x[0]):.3f} [{float(x[1]):.3f}, {float(x[2]):.3f}]: {x[4]}")
    csv_rows.append(dict(test="4a", kind="field", method="Shared-error-only", weighting=tag, ratio=float(x[0]), lo=float(x[1]), hi=float(x[2]),
                         lower=np.nan, fields=int(x[3]), holm=np.nan, verdict=x[4]))

say("\n-- test 1: rise in field-mean coverage D [95 % interval] (coverage does not depend on the weights) --")
for u, k, l in zip(h1_u, h1_k, h1_l):
    assert u[0] == k[0] == l[0], (u[0], k[0], l[0])
    for tag, x, head in zip(TAGS, (u, k, l), (f"{u[0]:12s}", " " * 12, " " * 12)):
        say(f"  {head} {tag:25s} D = {x[1]} [{x[2]}, {x[3]}] ({x[4]} fields)")
        csv_rows.append(dict(test="1", kind="field", method=x[0], weighting=tag, ratio=float(x[1]), lo=float(x[2]), hi=float(x[3]), lower=np.nan,
                             fields=int(x[4]), holm=np.nan, verdict="rises" if float(x[2]) > 0 else "does not rise"))
for tag, t in zip(TAGS, (t1_u, t1_k, t1_l)):
    say(f"  {'test 1':12s} {tag:25s} {t}")
    csv_rows.append(dict(test="1", kind="field", method="all four", weighting=tag, ratio=np.nan, lo=np.nan, hi=np.nan, lower=np.nan, fields=np.nan,
                         holm=np.nan, verdict=t))

say("\n-- against the unweighted results --")
for tag in TAGS[1:]:
    say(f"  {tag}: verdicts of test 5 that differ: " + ("none" if not differ[tag] else "; ".join(differ[tag])))
    say(f"  {'':{len(tag)}s}  weighted ratios above 1: " + ("none" if not above[tag] else "; ".join(above[tag])))
(OUT / "report9b.txt").write_text("\n".join(lines) + "\n")
pd.DataFrame(csv_rows).to_csv(OUT / "report9b.csv", index=False)
(OUT / "DONE").touch()
