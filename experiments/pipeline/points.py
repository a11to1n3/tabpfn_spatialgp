"""Point outputs of a set of splits (GPU; R for TabPFN-KpR's kriging): our TabPFN with cross-fitted residuals (point_model.our_model)
and TabPFN-KpR with the published settings (KpR) and with our TabPFN settings (KpR4) (point_model.kpr). Existing files are skipped, so
a crashed job is re-run with the identical command.

  python -m experiments.pipeline.points --set dev|first|final [--fields ...] [--min-test N] [--designs ...] [--reps ...] [--out DIR]
      default DIR: config.POINTS[set] ($SOILGP_RUN/points_dev, points_first, points). The development set is run with --min-test 1,
      because the fits behind the prior also use designs that leave fewer than 10 locations unrevealed; --designs and --reps are for
      small tests only.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one thread, as in the study's runs
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import json                                                                   # noqa: E402
import time                                                                   # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core.common import point_file, save_npz                      # noqa: E402
from experiments.core.point_model import kpr, our_model                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fields", nargs="*"); ap.add_argument("--out", default=None)
    ap.add_argument("--set", default="final", choices=("dev", "first", "final"))
    ap.add_argument("--min-test", type=int, default=None)                                 # 1 for the development splits of the prior fits
    ap.add_argument("--designs", nargs="*", help="small tests only"); ap.add_argument("--reps", nargs="*", type=int, help="small tests only")
    a = ap.parse_args()
    root = Path(a.out) if a.out else F.POINTS[a.set]
    for field, data, design, rep, tr in block_cases.iter_splits(a.set, a.fields, a.min_test):
        if (a.designs and design not in a.designs) or (a.reps and rep not in a.reps):
            continue
        te = np.setdiff1d(np.arange(len(data["coordinates"])), tr)
        for target in F.targets(field, a.set):
            t0 = time.monotonic(); y = data["Y"][:, F.C.TARGETS.index(target)]
            split_arrays = dict(train_rows=tr, test_rows=te, observed=y[te])
            path = lambda m: point_file(root, m, design, field, target, rep)            # noqa: E731
            if not path("ours").exists():
                arrays, info = our_model(field, data, design, rep, tr, te, target, "ours", ("Gbma",))
                save_npz(path("ours"), **split_arrays, **arrays, info=json.dumps(info))
            todo = [m for m in ("KpR", "KpR4") if not path(m).exists()]
            if todo:
                res, failed = kpr(data, tr, te, target, todo)
                for m, arrays in res.items():
                    meta = dict(field=field, target=target, design=design, rep=rep, model=m, kriging_failed=failed)
                    save_npz(path(m), **split_arrays, **arrays, metadata=json.dumps(meta))
            print(f"{a.set} {design} {field} {target} rep={rep} n={len(tr)} {time.monotonic() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
