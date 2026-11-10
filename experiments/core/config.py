"""The factor setting: the sets of splits, the comparators' candidate variants, and every fixed choice of their selection and factors.

Three sets of splits:
  dev    development splits: 14 development fields, random 5-90 %, replicates 15-19 (choice of the comparators' variants)
  first  factor-setting splits: 20 LimeSoDa fields at n = 5-100 (reps 40-49), Oxford, Meuse, Oxford without colour at n = 5-100 and
         5-90 % (reps 0-19); they set the comparators' factors (kappa = 1.10 for TabPFN-GP with these rows as stored)
  final  further splits of the same fields and designs, for the sensitivity checks: replicates 120-129 (LimeSoDa) and 120-139
         (Oxford, Meuse, Oxford without colour)
"""
import os
from pathlib import Path

import numpy as np

from experiments.core import datasets as C
from experiments.paths import REPO

RUN = Path(os.environ.get("SOILGP_RUN", REPO / "work" / "run"))   # run folder (point outputs, rows)
# Point outputs of each set (pipeline/points.py), one layout: <folder>/<model>/<design>/<field>/<target>/rep<r>.npz, model = ours, KpR, KpR4
POINTS = {"dev": RUN / "points_dev", "first": RUN / "points_first", "final": RUN / "points"}

# ------------------------------------------------------------------------------------------------------------------------- the sets
SETS = {
    "dev": [(C.DEV_FIELDS, C.RANDOM, range(15, 20))],
    "first": [(C.LIMESODA, C.N_DESIGNS, range(40, 50)), (("OXF", "MEUSE", "OXFNC"), C.N_DESIGNS + C.RANDOM, range(0, 20))],
    "final": [(C.LIMESODA, C.N_DESIGNS, range(120, 130)), (("OXF", "MEUSE", "OXFNC"), C.N_DESIGNS + C.RANDOM, range(120, 140))],
}
MIN_TEST = 10                                            # at least 10 unrevealed locations in every design

# ------------------------------------------------------------------------------------------------------------------------- tasks
def targets(field, set_name):
    """Properties analysed. Meuse: organic matter only, except in the first set's calibration rows, which keep Meuse's copies in the clay
    and pH slots, as stored (with them the factor rule gives kappa = 1.10)."""
    if field == "MEUSE" and set_name != "first":
        return ("SOC_target",)
    return C.TARGETS


PRIMARY_EXTERNAL = (("OXF", "SOC_target"), ("OXF", "Clay_target"), ("OXF", "pH_target"), ("MEUSE", "SOC_target"))
DENSE_FIELDS = ("B.204", "BB.250", "BB.72", "G.104", "G.150", "MG.112", "MGS.101", "BB.30_2", "MWP.36", "NSW.52", "SC.50", "SM.40",
                "LYD.112", "RUM.70")                     # the two SensRes fields have no spectra
DENSE_EXTERNAL = (("MEUSE", "SOC_target"), ("OXFNC", "SOC_target"), ("OXFNC", "Clay_target"), ("OXFNC", "pH_target"))
DUMMY_COORDS = ("SL.125", "UL.120")                     # released with dummy coordinates

# ------------------------------------------------------------------------------------------------------------------------- methods
KAPPA_OURS = 1.10                                     # the factor rule on the first set with its rows as stored; the study's 1.15 is the same rule with each task counted once
GATE_N = 25                                              # the GP's mean correction (and the plug-in variant's) from 25 samples (plan_development.md)
Z80 = 1.2815515655446004

# Candidate variants of each comparator, listed from simplest to most elaborate: the tie-break picks the first variant whose
# score is within 0.5 % of the best (local before cross-field variogram; kriging variance alone before adding TabPFN's v_T;
# in-sample before cross-fitted residuals; the plug-in analogue last).
GROUPS = {
    "TabPFN": ["TAB", "TAB1"],
    "TabPFN-KpR": ["KpR", "KpR4"],
    "Block RK": ["RK_ins_loc", "RK_oof_loc", "RK_ins_loc_v", "RK_oof_loc_v", "RK_ins_fix", "RK_oof_fix", "RK_ins_fix_v", "RK_oof_fix_v",
                 "PI_loc", "PI_fix"],
    "WH": ["WH_loc", "WH_fix"],
    "Sample mean": ["DB"],
}
COMPARATOR_VARIANTS = [v for g in GROUPS.values() for v in g]
TIE = 0.005
FACTOR_GRID = np.round(np.arange(0.50, 5.0001, 0.05), 2)  # calibration grid for every method
COVERAGE_TARGET = 0.80

# ------------------------------------------------------------------------------------------------------------------------- analysis
B = 10000
SEED = 0
E1_BAND = (0.75, 0.85)
ALPHA = 0.05
NONINF_MARGIN = 1.02
