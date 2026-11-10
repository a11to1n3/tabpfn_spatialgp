"""E7 of docs/exploratory/plan_E7-E8.md: the study's report with kappa = 1.10 for TabPFN-GP and the shared-error-only model (CPU).

  SOILGP_DATA=<work>/data python -m experiments.exploratory.kappa_110 <work>/rows <out>  -> <out>/report.txt
"""
import sys
from pathlib import Path

ROWS, OUT = sys.argv[1], Path(sys.argv[2])
from experiments.study import run as S                                        # noqa: E402

S.KAPPA = 1.10                                                                # the rule with the comparators' weighting
OUT.mkdir(parents=True, exist_ok=True)
S.stage_report(ROWS, str(OUT), S.evaluated())
