"""E8 of docs/exploratory/plan_E7-E8.md (GPU): TabPFN-GP rows of ten study fields with seeds 0-4 of TabPFN and of the fold split.

  SOILGP_DATA=<work>/data python -m experiments.exploratory.seeds_ten_fields <out>  -> <out>/seeds_<field>.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

OUT = Path(sys.argv[1])
from experiments.study import run as S                                        # noqa: E402

FIELDS = ("SOIL250", "SOJA98", "DK_EST", "DK_FAA", "PO_NE", "PO_RS", "NX_ON", "NX_CZK", "DR_AUZ", "DR_EPO")
DESIGNS, REPS, SEEDS = ("n5", "n20", "n50"), range(300, 305), (0, 1, 2, 3, 4)


def main():
    block_cases = S.setup(FIELDS)
    from experiments.sensitivity import designs as FD
    OUT.mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        path = OUT / f"seeds_{field}.csv"
        if path.exists():
            continue
        rows, t0 = [], time.monotonic()
        for f, data, design, rep, tr in block_cases.iter_splits("study", [field]):
            if design not in DESIGNS or rep not in REPS:
                continue
            xy = np.asarray(data["coordinates"], float); te = np.setdiff1d(np.arange(len(xy)), tr)
            for target in S.tasks(field):
                y = data["Y"][:, S.F.C.TARGETS.index(target)]
                for seed in SEEDS:
                    ours = FD.tabpfn_outputs(data, tr, te, target, seed=seed)
                    rows += FD.block_rows(dict(field=f, design=design, rep=rep, target=target, seed=seed), xy, tr, te, y, ours, kinds=("J1p",))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{field}: {len(rows)} rows, {time.monotonic() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
