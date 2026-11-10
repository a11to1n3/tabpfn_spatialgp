"""E19 of docs/exploratory/plan_E19.md (CPU): the CRPS with each mean weighted by its unrevealed share m_B / (m_B + n_B).

  SOILGP_DATA=<work>/data SOILGP_RUN=<work>/run python -m experiments.exploratory.weighted_crps <factor-setting folder> <work>/rows <work>/report/report.txt <out>
writes <out>/unweighted (the check) and <out>/weighted (versions, factors, the study's report), and <out>/report9.txt, report9.csv.
The study's scoring, selection and report code is used unchanged; only the CRPS column is multiplied by the weight when an area is scored.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A, ROWS, STORED, OUT = (Path(a) for a in sys.argv[1:5])
from experiments.study import run as S                                        # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.factor_setting import set_factors                            # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
FIXED = S.CONFIGS / "fixed"
ORIGINAL = scoring.score
H5 = re.compile(r"^H5 (field|zone)\s+TabPFN-GP / (.+?)\s+CRPS ratio ([\d.]+) \[([\d.]+), ([\d.]+)\], lower on (\d+) of (\d+) fields, "
                r"Wilcoxon p = (\S+), Holm (\S+): (.+)$")
H4 = re.compile(r"^H4\(a\) .*CRPS ratio to TabPFN-GP ([\d.]+) \[([\d.]+), ([\d.]+)\] \((\d+) fields\): (.+)$")
lines, csv_rows = [], []


def say(s=""):
    lines.append(s); print(s, flush=True)


def weighted(rows, factors=None):
    r = ORIGINAL(rows, factors)
    r["crps"] = r["crps"].to_numpy() * (r.mB.to_numpy(float) / (r.mB.to_numpy(float) + r.nB.to_numpy(float)))
    return r


def run(name, score):
    """Versions on the development splits, factors and CRPS versions on the factor-setting data, then the study's report."""
    out = OUT / name; fixed = out / "fixed"; fixed.mkdir(parents=True, exist_ok=True)
    scoring.score = score
    set_factors.select(str(A), str(fixed))
    set_factors.calibrate(str(A), str(fixed / "selection.json"), str(fixed))
    S.CONFIGS = out
    S.stage_report(str(ROWS), str(out / "report"), S.evaluated())
    comp, factors, crps_f, names = S.methods()
    raw = S.load_rows(str(ROWS), S.evaluated())
    common = S.area_sets(raw, comp, factors, crps_f)[1]
    j = common[(common.variant == "J1p") & (common.block == "field")]
    return float(j[j.n <= 20].crps.sum() / j.crps.sum()), out


def comparators(folder):
    return json.loads((folder / "calibration.json").read_text())["comparators"]


def tests(path):
    t = path.read_text().splitlines()
    return [m.groups() for m in map(H5.match, t) if m], next(m.groups() for m in map(H4.match, t) if m)


# ------------------------------------------------------------------------------------------------------------------- checks
for pattern, folder in (("rows_dev_*.csv", A), ("rows_first_*.csv", A), ("rows_study_*.csv", ROWS)):
    files = sorted(folder.glob(pattern)); assert files, pattern
    for p in files:
        r = pd.read_csv(p, usecols=["block", "n", "m", "mB", "nB"], low_memory=False)
        f = r[r.block == "field"]; assert (f.nB == f.n).all() and (f.mB == f.m).all(), ("check 2", p.name)
        w = r.mB / (r.mB + r.nB); assert ((w > 0) & (w <= 1)).all(), ("check 2: weights", p.name)
print("check 2 passed: field rows have n_B = n and m_B = m; every weight lies in (0, 1]", flush=True)

share_u, folder_u = run("unweighted", ORIGINAL)
assert json.loads((FIXED / "selection.json").read_text()) == json.loads((folder_u / "fixed" / "selection.json").read_text()), "check 1: versions"
stored_cal, new_cal = comparators(FIXED), comparators(folder_u / "fixed")
for g in stored_cal:
    for k in ("variant", "factor", "crps_version"):
        assert stored_cal[g][k] == new_cal[g][k], ("check 1", g, k, stored_cal[g][k], new_cal[g][k])
assert (folder_u / "report" / "report.txt").read_bytes() == STORED.read_bytes(), "check 1: report differs from the study's"
print("check 1 passed: versions, factors and report identical to the study's", flush=True)

# ------------------------------------------------------------------------------------------------------------------- weighted
share_w, folder_w = run("weighted", weighted)
cal_w = comparators(folder_w / "fixed")
h5_u, h4_u = tests(folder_u / "report" / "report.txt")
h5_w, h4_w = tests(folder_w / "report" / "report.txt")

say("=== E19 (docs/exploratory/plan_E19.md; specified after the study's results had been seen; exploratory) ===")
say("CRPS of each mean multiplied by m_B / (m_B + n_B); TabPFN-GP with kappa = 1.15; the comparators' versions, factors and CRPS versions set by the same rules with the weighted CRPS.")
say("\n-- comparators: version, factor, CRPS version; the study's | with the weighted CRPS --")
for g in stored_cal:
    a, b = stored_cal[g], cal_w[g]
    say(f"  {g:12s} {a['variant']}, factor {a['factor']:.2f}, CRPS {a['crps_version']} | "
        f"{b['variant']}, factor {b['factor']:.2f}, CRPS {b['crps_version']}")
say("\n-- test 5: CRPS ratio TabPFN-GP / method [95 % interval], fields lower, Holm p, verdict --")
changed, broken, above = [], [], []
for u, w in zip(h5_u, h5_w):
    assert u[:2] == w[:2], (u[:2], w[:2])
    say(f"  {u[0]:5s} {u[1]:12s} unweighted {float(u[2]):.3f} [{float(u[3]):.3f}, {float(u[4]):.3f}], {u[5]}/{u[6]}, Holm {float(u[8]):.3g}: {u[9]}")
    say(f"  {'':5s} {'':12s} weighted   {float(w[2]):.3f} [{float(w[3]):.3f}, {float(w[4]):.3f}], {w[5]}/{w[6]}, Holm {float(w[8]):.3g}: {w[9]}")
    for tag, x in (("unweighted", u), ("weighted", w)):
        csv_rows.append(dict(test="5", kind=x[0], method=x[1], weighting=tag, ratio=float(x[2]), lo=float(x[3]), hi=float(x[4]), lower=int(x[5]),
                             fields=int(x[6]), holm=float(x[8]), verdict=x[9]))
    if u[9] != w[9]:
        changed.append(f"{u[0]} / {u[1]}: {u[9]} -> {w[9]}")
    if w[9] not in ("better", "no worse"):
        broken.append(f"{w[0]} / {w[1]}")
    if float(w[2]) > 1:
        above.append(f"{w[0]} / {w[1]} ({float(w[2]):.3f})")
say("\n-- test 4a: CRPS ratio shared-error-only / TabPFN-GP --")
say(f"  unweighted {float(h4_u[0]):.3f} [{float(h4_u[1]):.3f}, {float(h4_u[2]):.3f}]: {h4_u[4]}")
say(f"  weighted   {float(h4_w[0]):.3f} [{float(h4_w[1]):.3f}, {float(h4_w[2]):.3f}]: {h4_w[4]}")
for tag, x in (("unweighted", h4_u), ("weighted", h4_w)):
    csv_rows.append(dict(test="4a", kind="field", method="Shared-error-only", weighting=tag, ratio=float(x[0]), lo=float(x[1]), hi=float(x[2]),
                         lower=np.nan, fields=int(x[3]), holm=np.nan, verdict=x[4]))
say(f"\n-- share of TabPFN-GP's summed field-mean CRPS from 5 to 20 samples: unweighted {share_u:.3f}, weighted {share_w:.3f} --")
say("\n-- reading (fixed before running, docs/exploratory/plan_E19.md) --")
say("  verdicts of test 5: " + ("all unchanged" if not changed else "changed: " + "; ".join(changed)))
say("  contrasts no longer met: " + ("none" if not broken else "; ".join(broken)))
say("  weighted ratios above 1: " + ("none" if not above else "; ".join(above)))
say(f"  test 4a: {h4_u[4]} -> {h4_w[4]}")
(OUT / "report9.txt").write_text("\n".join(lines) + "\n")
pd.DataFrame(csv_rows).to_csv(OUT / "report9.csv", index=False)
(OUT / "DONE").touch()
