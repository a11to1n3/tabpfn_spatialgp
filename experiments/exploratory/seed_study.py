"""E12 of docs/exploratory/plan_E9-E13.md: the study's predictions and area rows with another seed of TabPFN and of the fold split.

  SOILGP_DATA=<work>/data SOILGP_RUN=<work>/run_seed<s> python -m experiments.exploratory.seed_study points <seed> <field>
  SOILGP_DATA=<work>/data SOILGP_RUN=<work>/run_seed<s> python -m experiments.exploratory.seed_study rows <seed> <out> [workers]

The study's code is used unchanged; only the seed is set, through tabpfn_spatialgp.settings.RANDOM_STATE (the fold split and the
point GP) and the default seed of common.tabpfn (every TabPFN fit: ours, the fold models and TabPFN-KpR). Seed 0 is the study.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                                    # noqa: E402

stage, SEED = sys.argv[1], int(sys.argv[2])
from experiments.study import run as S                                        # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402
from experiments.core import common                                           # noqa: E402

S_.RANDOM_STATE = SEED
d = list(common.tabpfn.__defaults__); d[2] = SEED; common.tabpfn.__defaults__ = tuple(d)   # (n_estimators, target_transform, seed)

if stage == "points":
    S.stage_points((sys.argv[3],), None)
elif stage == "rows":
    S.stage_rows(S.evaluated(), sys.argv[3], None, int(sys.argv[4]) if len(sys.argv) > 4 else 8)
else:
    raise SystemExit(f"unknown stage {stage}")
