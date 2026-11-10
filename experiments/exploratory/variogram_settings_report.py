"""E18 of docs/exploratory/plan_E18.md: summary of Block RK and WH with their variogram settings from fewer development fields.

  SOILGP_DATA=<work>/data python -m experiments.exploratory.variogram_settings_report <stored rows> <E18 folder> <out>
writes <out>/report8.txt and <out>/report8.csv from the folders written by run_variogram_settings.sh.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROWS, E18, OUT = (Path(a) for a in sys.argv[1:4])
from experiments.study import run as S                                        # noqa: E402

CONFIGS = S.CONFIGS
OUT.mkdir(parents=True, exist_ok=True)
lines, csv_rows = [], []
GROUPS = ("Block RK", "WH")
NAMES = {"ALL": "all 14 development fields (as in the study)", "X2": "without SL.125 and UL.120", "X3": "without SL.125, UL.120 and SP.231"}
H1 = re.compile(r"^H1 (\S+(?: \S+)?)\s+field-mean D = ([+-][\d.]+), 95 % \[([+-][\d.]+), ([+-][\d.]+)\]")
H5 = re.compile(r"^H5 (field|zone)\s+TabPFN-GP / (.+?)\s+CRPS ratio ([\d.]+) \[([\d.]+), ([\d.]+)\], lower on (\d+) of (\d+) fields, "
                r"Wilcoxon p = (\S+), Holm (\S+): (.+)$")


def say(s=""):
    lines.append(s); print(s, flush=True)


def rec(name, group, what, kind, est, lo=np.nan, hi=np.nan, note=""):
    csv_rows.append(dict(set=name, comparator=group, what=what, kind=kind, est=est, lo=lo, hi=hi, note=note))


def fmt(e, ci):
    return f"{e:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def tests(path):
    t = path.read_text().splitlines()
    return ([m.groups() for m in map(H1.match, t) if m], next((x for x in t if x.startswith("H1 (")), ""),
            [m.groups() for m in map(H5.match, t) if m])


say("=== E18 (docs/exploratory/plan_E18.md; specified after the study's results had been seen; exploratory) ===")
verdicts = {}
for name in ("ALL", "X2", "X3"):
    root = CONFIGS if name == "ALL" else E18 / name
    rows_dir = ROWS if name == "ALL" else E18 / name / "rows"
    rep = (E18 / "check3" / "report" if name == "ALL" else E18 / name / "report") / "report.txt"
    S.CONFIGS = root
    comp, factors, crps_f, names = S.methods()
    cal = json.loads((root / "fixed" / "calibration.json").read_text())["comparators"]
    raw = S.load_rows(str(rows_dir), S.evaluated())
    cov, common, db, pair_o1 = S.area_sets(raw, comp, factors, crps_f)
    shapes = json.loads((CONFIGS / "fixed" / "shapes.json").read_text() if name == "ALL" else (E18 / f"shapes_{name}.json").read_text())
    say(f"\n-- {name}: medians over {NAMES[name]} --")
    for lab, c in (("Block RK, in-sample residuals (r_ins)", "r_ins"), ("Block RK, held-out errors (r_oof)", "r_oof"), ("WH (z_oof)", "z_oof")):
        say(f"  {lab}: range/extent {shapes[c]['range_rel']:.3f}, nugget share {shapes[c]['nugget_share']:.3f} ({shapes[c]['fits']} fits)")
    say(f"  areas scored by all methods: {(common[common.variant == 'J1p'].block == 'field').sum()} field and "
        f"{(common[common.variant == 'J1p'].block != 'field').sum()} zone means")
    h1, h1_line, h5 = tests(rep)
    for g in GROUPS:
        v = cal[g]["variant"]
        rec(name, g, "range_rel", "", shapes["z_oof" if g == "WH" else "r_oof" if "oof" in v else "r_ins"]["range_rel"], note=v)
        rec(name, g, "nugget_share", "", shapes["z_oof" if g == "WH" else "r_oof" if "oof" in v else "r_ins"]["nugget_share"], note=v)
        rec(name, g, "factor", "", cal[g]["factor"], note=cal[g]["crps_version"])
        say(f"  {g}: version {v}; factor {cal[g]['factor']:.2f}; CRPS {cal[g]['crps_version']}; "
            f"failed {int(raw[raw.variant == v].failed.sum())}/{int((raw.variant == v).sum())} areas")
        for k in ("field", "zone"):
            e, ci, nf = S.coverage(common, v, k); d, b, nd = S.rise(common, v, k); lo, hi = np.percentile(b, [2.5, 97.5])
            row = next(r for r in h5 if r[0] == k and r[1] == g)
            ratio, rlo, rhi, verdict = float(row[2]), float(row[3]), float(row[4]), row[9]
            say(f"    {k}: coverage {fmt(e, ci)}; D {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; CRPS ratio TabPFN-GP / {g} {fmt(ratio, (rlo, rhi))}, "
                f"lower on {row[5]} of {row[6]} fields, Holm {row[8]}: {verdict}")
            rec(name, g, "coverage", k, e, *ci, note=v); rec(name, g, "D", k, d, lo, hi, note=v)
            rec(name, g, "crps_ratio", k, ratio, rlo, rhi, note=verdict)
            verdicts[(name, g, k)] = verdict
    say("  test 1: " + "; ".join(f"{r[0]} D {r[1]} [{r[2]}, {r[3]}]" for r in h1))
    say("  " + h1_line)
    say("  test 5, all ten contrasts: " + "; ".join(f"{r[0]} / {r[1]}: {r[9]}" for r in h5))
    verdicts[(name, "test1")] = h1_line.endswith(": met")
    verdicts[(name, "test5")] = all(r[9] in ("better", "no worse") for r in h5)

say("\n-- reading (fixed before running, docs/exploratory/plan_E18.md) --")
for name in ("X2", "X3"):
    change = [f"{g} {k}: {verdicts[('ALL', g, k)]} -> {verdicts[(name, g, k)]}" for g in GROUPS for k in ("field", "zone")
              if verdicts[(name, g, k)] != verdicts[("ALL", g, k)]]
    say(f"  {name}: test 1 {'met' if verdicts[(name, 'test1')] else 'NOT met'}; test 5 {'met' if verdicts[(name, 'test5')] else 'NOT met'}; "
        f"verdicts for Block RK and WH {'unchanged' if not change else 'changed: ' + '; '.join(change)}")

(OUT / "report8.txt").write_text("\n".join(lines) + "\n")
pd.DataFrame(csv_rows).to_csv(OUT / "report8.csv", index=False)
