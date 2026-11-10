"""E18 of docs/exploratory/plan_E18.md (CPU and R): Block RK and WH with their variogram settings from fewer development fields.

  python -m experiments.exploratory.variogram_settings settings <fits.csv> <OUT>
      medians over all 14 fields (ALL), X2 and X3 -> OUT/shapes_<name>.json; check 1 of the plan
  SOILGP_DATA=.. SOILGP_RUN=.. python -m experiments.exploratory.variogram_settings rows <dev|first|study> <shapes.json> <out> <workers> [field ...]
      rows of the variants that use the settings, from the saved point outputs -> <out>/fix_<set>_<field>.csv
  python -m experiments.exploratory.variogram_settings compare <fix dir> <stored rows dir>
      check 2 of the plan: every file of <fix dir> against the stored rows
  python -m experiments.exploratory.variogram_settings merge <fix dir> <stored rows dir> <dev|first|study> <out>
      the stored rows of the set with those variants replaced -> <out>/rows_<set>_<field>.csv
  SOILGP_DATA=<work>/data python -m experiments.exploratory.variogram_settings report <rows dir> <folder holding fixed/calibration.json> <out>
      the study's report (study/run.py) with these comparators' versions and factors
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

CMD = sys.argv[1]
from experiments.study import run as S                                        # noqa: E402

F = S.F
DROP = {"ALL": [], "X2": ["SL.125", "UL.120"], "X3": ["SL.125", "UL.120", "SP.231"]}
EXPECTED = {"X2": {"r_ins": (0.075, 0.108), "z_oof": (0.069, 0.151)}, "X3": {"r_ins": (0.077, 0.091), "z_oof": (0.074, 0.029)}}
FIX = ("RK_ins_fix", "RK_ins_fix_v", "RK_oof_fix", "RK_oof_fix_v", "PI_fix", "WH_fix")
KEYS = ["field", "design", "rep", "target", "block", "variant"]


def settings(fits_csv, out):
    """Medians as pipeline/block_rows.py computes them, over the fits of each set's fields."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(fits_csv, float_precision="round_trip"); ok = fits[fits.ok == True]                  # noqa: E712
    for name, drop in DROP.items():
        keep = ~ok.field.isin(drop); shapes = {}
        for c, g in ok[keep].groupby("column"):
            shapes[c] = {"range_rel": float(np.median(g.range_rel)), "nugget_share": float(np.median(g.nugget_share)), "fits": int(len(g)),
                         "failed": int(((fits.column == c) & (fits.ok != True) & ~fits.field.isin(drop)).sum())}      # noqa: E712
        (out / f"shapes_{name}.json").write_text(json.dumps(shapes, indent=1) + "\n")
        print(name, json.dumps(shapes), flush=True)
    stored = json.loads((S.CONFIGS / "fixed" / "shapes.json").read_text())
    new = json.loads((out / "shapes_ALL.json").read_text())
    assert stored == new, ("check 1: the medians over all 14 fields differ from fixed/shapes.json", stored, new)
    for name, exp in EXPECTED.items():
        got = json.loads((out / f"shapes_{name}.json").read_text())
        for c, (r, nu) in exp.items():
            assert round(got[c]["range_rel"], 3) == r and round(got[c]["nugget_share"], 3) == nu, ("check 1", name, c, got[c])
    print("check 1 passed", flush=True)


def field_job(job):
    set_name, field, shapes, out = job
    from experiments.pipeline import block_rows
    from experiments.core import block_cases
    t0 = time.monotonic(); rows, missing = [], 0
    for key, case in block_rows.cases(set_name, field):
        if case is None:
            missing += 1; continue
        rows += [r for r in block_cases.case_rows(case, shapes, ours=False) if r["variant"] in FIX]
    path = Path(out) / f"fix_{set_name}_{field}.csv"; tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, path)
    return field, len(rows), missing, time.monotonic() - t0


def rows(set_name, shapes_file, out, workers, only):
    if set_name == "study":
        block_cases = S.setup(S.evaluated())
        fields = list(S.evaluated())
    else:
        from experiments.core import block_cases
        fields = sorted({f for group, _, _ in F.SETS[set_name] for f in group})
    from experiments.pipeline import block_rows                               # noqa: F401
    fields = [f for f in fields if not only or f in only]
    original = block_cases.run_r

    def fix_only(mode, s, tr, te, cols, params=None):
        """The local (automap) variants do not use the settings: their stored rows are kept, so their R process is skipped."""
        if params is not None and params.get("parts") == ["loc"]:
            raise RuntimeError("not needed")
        return original(mode, s, tr, te, cols, params)

    block_cases.run_r = fix_only
    shapes = json.loads(Path(shapes_file).read_text())
    Path(out).mkdir(parents=True, exist_ok=True)
    jobs = [(set_name, f, shapes, str(out)) for f in fields if not (Path(out) / f"fix_{set_name}_{f}.csv").exists()]
    print(f"{set_name}: {len(jobs)} of {len(fields)} fields to do", flush=True)
    if jobs:
        with Pool(min(int(workers), len(jobs))) as pool:
            for field, n, missing, sec in pool.imap_unordered(field_job, jobs):
                print(f"{set_name} {field}: {n} rows, {missing} cases without point outputs, {sec:.0f} s", flush=True)
                assert missing == 0, (field, missing)


def stored_of(path, stored_dir):
    set_name, field = path.stem[len("fix_"):].split("_", 1)
    return set_name, field, Path(stored_dir) / f"rows_{set_name}_{field}.csv"


def compare(fix_dir, stored_dir):
    """Check 2: same rows, same failures, M and V within 1e-9."""
    files = sorted(Path(fix_dir).glob("fix_*.csv")); assert files, "no files"
    for p in files:
        set_name, field, sp = stored_of(p, stored_dir)
        new = pd.read_csv(p, low_memory=False)
        old = pd.read_csv(sp, low_memory=False); old = old[old.variant.isin(FIX)]
        assert len(new) == len(old), ("check 2: row counts", p.name, len(new), len(old))
        m = old.merge(new, on=KEYS, suffixes=("_o", "_n"), validate="one_to_one"); assert len(m) == len(old), ("check 2: keys", p.name)
        assert (m.failed_o.astype(bool) == m.failed_n.astype(bool)).all(), ("check 2: failures", p.name)
        good = ~m.failed_o.astype(bool)
        dm, dv = (np.abs(m.loc[good, f"{c}_o"] - m.loc[good, f"{c}_n"]).max() for c in ("M", "V"))
        print(f"{p.name}: {len(m)} rows, {int((~good).sum())} failed in both, max |dM| = {dm:.2e}, max |dV| = {dv:.2e}", flush=True)
        assert dm <= 1e-9 and dv <= 1e-9, ("check 2: values differ", p.name, dm, dv)
    print("check 2 passed", flush=True)


def merge(fix_dir, stored_dir, set_name, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(fix_dir).glob(f"fix_{set_name}_*.csv")); assert files, "no files"
    stored = sorted(Path(stored_dir).glob(f"rows_{set_name}_*.csv")); assert len(files) == len(stored), (len(files), len(stored))
    for p in files:
        _, field, sp = stored_of(p, stored_dir)
        old = pd.read_csv(sp, low_memory=False); new = pd.read_csv(p, low_memory=False)
        was = old[old.variant.isin(FIX)]
        chk = was.merge(new, on=KEYS, suffixes=("_o", "_n"), validate="one_to_one")
        assert len(chk) == len(was) == len(new), ("merge: rows differ", p.name, len(chk), len(was), len(new))
        assert np.allclose(chk.obs_o, chk.obs_n, atol=1e-9, rtol=0), ("merge: observed means differ", p.name)
        pd.concat([old[~old.variant.isin(FIX)], new[old.columns]], ignore_index=True).to_csv(out / sp.name, index=False)
    if set_name == "study":                                                   # the shared-error-only rows, read by study.run.load_rows
        for p in Path(stored_dir).glob("*.csv"):
            if not p.name.startswith("rows_") and not (out / p.name).exists():
                (out / p.name).symlink_to(p.resolve())
    print(f"merged {len(files)} fields of {set_name} -> {out}", flush=True)


def report(rows_dir, cal_dir, out):
    S.CONFIGS = Path(cal_dir)                                                   # study.run.methods reads CONFIGS/fixed/calibration.json
    S.stage_report(rows_dir, out, S.evaluated())


if __name__ == "__main__":
    a = sys.argv[2:]
    if CMD == "settings":
        settings(a[0], a[1])
    elif CMD == "rows":
        rows(a[0], a[1], a[2], a[3], a[4:])
    elif CMD == "compare":
        compare(a[0], a[1])
    elif CMD == "merge":
        merge(a[0], a[1], a[2], a[3])
    elif CMD == "report":
        report(a[0], a[1], a[2])
    else:
        sys.exit(__doc__)
