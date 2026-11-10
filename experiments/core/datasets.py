"""Fields, properties and designs of the study, and the folders of the data.

Paths come from environment variables so that the code runs anywhere:
    SOILGP_DATA   folder with one <field>.npz per field (written by data/export_data.py; default: <repo>/work/data)
The sets of splits (development, first, final) are defined in config.py.
"""
import os
from pathlib import Path

from experiments.paths import REPO

DATA = Path(os.environ.get("SOILGP_DATA", REPO / "work" / "data"))

TARGETS = ("SOC_target", "Clay_target", "pH_target")    # organic matter takes the SOC slot where SOC is not measured
QUANTILES = (0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999)   # saved predictive quantiles; [1] = 0.1 and [9] = 0.9 bound the 80 % interval

# --- fields -----------------------------------------------------------------------------------------------------------------------------
KPR_FIELDS = ("BB.250", "BB.72", "G.150", "MG.112", "SA.112", "RP.62")          # the six fields of the KpR paper (Schmidinger et al., 2026)
OTHER_FIELDS = ("B.204", "G.104", "MGS.101", "SP.231", "SL.125", "UL.120", "H.138", "SC.93")
UNSHARED = ("RP.62", "SA.112")                                                   # KpR fields whose coordinates cannot be shared: not used
SENSRES = ("LYD.112", "RUM.70")                                                  # SensRes fields (Lithuania, CC BY 4.0), development fields
DEV_FIELDS = tuple(f for f in KPR_FIELDS + OTHER_FIELDS if f not in UNSHARED) + SENSRES   # the 14 development fields
HELD_BACK = ("BB.30_2", "MG.44", "MWP.36", "NSW.52", "SC.50", "SM.40")           # the six held-back fields
LIMESODA = DEV_FIELDS + HELD_BACK                                                # the 20 fields: 18 LimeSoDa fields and the two SensRes fields
EXTERNAL = ("OXF", "MEUSE", "OXFNC")                                             # Oxford, Meuse; OXFNC = Oxford without colour covariates
EXTERNAL_TASKS = (("OXF", "SOC_target"), ("OXF", "Clay_target"), ("OXF", "pH_target"), ("MEUSE", "SOC_target"))   # primary external tasks

# --- designs (training sets; every other sample of the field is a test point) ----------------------------------------------------------
N_DESIGNS = tuple(f"n{n}" for n in (5, 10, 15, 20, 30, 50, 75, 100))             # N random samples (only if N <= field size - 10)
RANDOM = tuple(f"random{p}" for p in (5, 10, 20, 30, 50, 70, 90))                # p % random samples
CLUSTER = ("cluster20", "cluster40")                                             # p % in 4 compact clusters
FOLDS = tuple(f"fold{k}" for k in range(1, 11))                                  # the KpR authors' published 10-fold CV

