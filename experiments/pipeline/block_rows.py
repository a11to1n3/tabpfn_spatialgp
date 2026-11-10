"""Block rows of a set, and the cross-field variogram shapes. CPU and R only; reads the saved point outputs.

  python -m experiments.pipeline.block_rows shapes --out DIR
      exponential fits on the development splits with n >= 50 -> DIR/shape_fits.csv and DIR/shapes.json (median range/extent and
      nugget share per residual type)
  python -m experiments.pipeline.block_rows rows --set dev|first|final --shapes DIR/shapes.json --out DIR [--fields ...] [--workers 16]
      one CSV per field: DIR/rows_<set>_<field>.csv (block_cases.case_rows)
Every Rscript call is single-threaded; fields run in parallel.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import json                                                                   # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402


FILTER = {}                                                                   # --designs / --reps: for small tests only


def cases(set_name, field):
    for f, data, design, rep, tr in block_cases.iter_splits(set_name, [field]):
        if (FILTER.get('designs') and design not in FILTER['designs']) or (FILTER.get('reps') and rep not in FILTER['reps']):
            continue
        for target in F.targets(f, set_name):
            y = data["Y"][:, F.C.TARGETS.index(target)]
            if not np.all(np.isfinite(y)):
                continue
            case = block_cases.load_case(set_name, f, data, design, rep, tr, target)
            yield (f, design, rep, target), case


def field_shapes(field):
    out = []
    for key, case in cases("dev", field):
        if case is None or len(case["tr"]) < 50:
            continue
        out += block_cases.shape_fits(case)
    return out


def field_rows(job):
    set_name, field, shapes, out_dir, flt = job
    FILTER.update(flt)
    t0 = time.monotonic(); rows, missing = [], []
    for key, case in cases(set_name, field):
        if case is None:
            missing.append(key); continue
        rows += block_cases.case_rows(case, shapes, ours=(set_name != "dev"))
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"rows_{set_name}_{field}.csv", index=False)
    return field, len(rows), missing, time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["shapes", "rows"])
    ap.add_argument("--set", choices=list(F.SETS)); ap.add_argument("--shapes"); ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="*"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--designs", nargs="*", help="small tests only"); ap.add_argument("--reps", nargs="*", type=int, help="small tests only")
    a = ap.parse_args(); flt = {"designs": a.designs, "reps": a.reps}; FILTER.update(flt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.stage == "shapes":
        fields = a.fields or list(F.C.DEV_FIELDS)
        with Pool(min(a.workers, len(fields))) as pool:
            fits = pd.DataFrame([r for res in pool.map(field_shapes, fields) for r in res])
        fits.to_csv(out / "shape_fits.csv", index=False)
        ok = fits[fits.ok == True]                                            # noqa: E712
        shapes = {c: {"range_rel": float(np.median(g.range_rel)), "nugget_share": float(np.median(g.nugget_share)), "fits": int(len(g)),
                      "failed": int(((fits.column == c) & (fits.ok != True)).sum())}                                     # noqa: E712
                  for c, g in ok.groupby("column")}
        (out / "shapes.json").write_text(json.dumps(shapes, indent=1) + "\n")
        print(json.dumps(shapes, indent=1))
        return
    shapes = json.loads(Path(a.shapes).read_text()) if a.shapes else None
    fields = a.fields or sorted({f for group, _, _ in F.SETS[a.set] for f in group})
    jobs = [(a.set, f, shapes, str(out), flt) for f in fields]
    with Pool(min(a.workers, len(jobs))) as pool:
        for field, n, missing, sec in pool.imap_unordered(field_rows, jobs):
            print(f"{a.set} {field}: {n} rows, {len(missing)} cases without point outputs, {sec:.0f} s", flush=True)
            if missing:
                print("   missing:", missing[:5], "..." if len(missing) > 5 else "", flush=True)


if __name__ == "__main__":
    main()
