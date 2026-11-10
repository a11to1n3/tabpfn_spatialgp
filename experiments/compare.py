"""Compare a reproduction with the outputs behind the paper.

  python -m experiments.compare factor-setting <WORK>
      WORK/factor_setting/{shapes,selection,calibration}.json against configs/fixed/, the stored outputs of the factor setting that
      the study uses: the shapes, the selected variants, every factor (kappa included) and CRPS version must agree; the coverages
      behind the factors are reported (see README.md: other GPUs can move them slightly)
  python -m experiments.compare reports <WORK>
      the dry run of the factor setting and the study's export log and report against expected/factor_setting/ and expected/study/,
      line by line, with the folders written as <WORK> and <REPO>; also the derived prior (WORK/prior_check.txt)
Exit status 0 if everything agrees.
"""
import json
import math
import sys
from pathlib import Path

from experiments.paths import CONFIGS, REPO

FIXED = CONFIGS / "fixed"
REPORTS = {                                    # <expected file>  <-  <WORK>/<path>
    "expected/factor_setting/report_first.txt": "factor_setting/report_first.txt",
    "expected/study/export_log.csv": "study/csv/export_log.csv",
    "expected/study/report.txt": "study/report/report.txt",
    "expected/study/coverage_by_n.csv": "study/report/coverage_by_n.csv",
}


def same(a, b, path=""):
    """Differences between two JSON values: numbers to a relative 1e-9, everything else exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: keys {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
        return out + [d for k in sorted(set(a) & set(b)) for d in same(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        ok = math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))
        return [] if ok else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def factor_setting(work):
    bad = 0
    for name in ("shapes.json", "selection.json"):
        diffs = same(json.loads((work / "factor_setting" / name).read_text()), json.loads((FIXED / name).read_text()))
        bad += bool(diffs)
        print(f"{name}: " + ("identical to fixed/" if not diffs else f"{len(diffs)} differences, e.g. {diffs[:3]}"))
    got, ref = (json.loads(p.read_text()) for p in (work / "factor_setting" / "calibration.json", FIXED / "calibration.json"))
    decision = lambda c: ({g: (v["variant"], v["factor"], v["crps_version"]) for g, v in c["comparators"].items()}, c["check"]["J1p_factor"])  # noqa: E731
    same_decision = decision(got) == decision(ref); bad += not same_decision
    diffs = same(got, ref); largest = max((abs(float(d.split(": ")[1].split(" != ")[0]) - float(d.split(" != ")[1])) for d in diffs), default=0.0)
    print("calibration.json: the selected variants, every factor, kappa and the CRPS versions are "
          + ("identical to fixed/" if same_decision else "DIFFERENT from fixed/")
          + (f"; {len(diffs)} coverages behind them differ, by at most {largest:.1e}" if diffs else "; every number identical"))
    return bad


def normalised(text, work):
    return text.replace(str(work), "<WORK>").replace(str(REPO), "<REPO>")


def reports(work):
    bad = 0
    for name, rel in REPORTS.items():
        got_path = work / rel
        if not got_path.exists():
            print(f"{name}: MISSING ({got_path})"); bad += 1; continue
        got, exp = normalised(got_path.read_text(), work).splitlines(), (REPO / name).read_text().splitlines()
        diff = [i for i, (x, y) in enumerate(zip(got, exp)) if x != y] + ([min(len(got), len(exp))] if len(got) != len(exp) else [])
        bad += bool(diff)
        print(f"{name}: " + ("identical" if not diff else f"{len(diff)} lines differ, first at line {diff[0] + 1}"))
    prior = (work / "prior_check.txt").read_text() if (work / "prior_check.txt").exists() else ""
    ok = "priors.json: identical" in prior and "BLOCK_PRIOR: identical" in prior
    bad += not ok
    print("derived prior: " + ("identical to priors.json and the package's BLOCK_PRIOR" if ok else "DIFFERENT or missing"))
    return bad


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("factor-setting", "reports"):
        sys.exit(__doc__)
    work = Path(sys.argv[2]).resolve()
    sys.exit(1 if (factor_setting(work) if sys.argv[1] == "factor-setting" else reports(work)) else 0)
