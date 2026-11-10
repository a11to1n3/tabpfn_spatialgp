#!/usr/bin/env bash
# The study of docs/PROTOCOL.md, run once: fetch and check the sources, export (data rules and covariates), predictions, area rows, report.
#   bash experiments/study/run_study.sh all                      every stage in order
#   bash experiments/study/run_study.sh fetch export data ...    the named stages (fetch export data points rows report), in the order given
#   bash experiments/study/run_study.sh dryrun                   code check on the development field BB.72 (two repeats; outputs not
#                                                                reported); needs DRY_DATA, the folder holding BB.72.npz (reproduce.sh,
#                                                                stage data)
# Settings: WORK (default ./work_study), everything this script writes; PY (default python), the Python of environment/requirements.txt;
# JOBS (default 6), parallel GPU jobs for the predictions; SOILGP_RLIB, the R library (gstat, automap) if not the default.
# Predictions that already exist are skipped, so an interrupted stage is resumed by running it again.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(mkdir -p "${WORK:-$REPO/work_study}" && cd "${WORK:-$REPO/work_study}" && pwd)
PY=${PY:-python}
case "$PY" in */*) PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY");; esac     # every module runs from the repository root
JOBS=${JOBS:-6}
HERE=$REPO/experiments/study
RUN="-m experiments.study.run"
LOGS=$WORK/logs
mkdir -p "$LOGS"
export SOILGP_DATA=$WORK/data SOILGP_RUN=$WORK/run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
[ -z "${DRY_DATA:-}" ] || DRY_DATA=$(cd "$DRY_DATA" && pwd)
cd "$REPO"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOGS/run_study.log"; }
fields() { "$PY" -c "from experiments.study import run as s; print(' '.join(s.evaluated()))"; }

stage_fetch() { say "fetch: sources and checksums"; PY=$PY bash "$HERE/fetch.sh" "$WORK/sources" > "$LOGS/fetch.log" 2>&1; tail -1 "$LOGS/fetch.log"; }
stage_export() {
  say "export: data rules and covariates"
  Rscript "$HERE/export.R" "$WORK/sources" "$WORK/csv" > "$LOGS/export.log" 2>&1
  grep -c "covariates (" "$LOGS/export.log" | xargs echo "fields written:"
}
stage_data() { say "data"; "$PY" $RUN data --csv "$WORK/csv" > "$LOGS/data.log" 2>&1; tail -1 "$LOGS/data.log"; }
stage_points() {
  local fs; fs=$(fields)
  say "points: $(echo $fs | wc -w) fields, $JOBS at a time"
  printf '%s\n' $fs | xargs -P "$JOBS" -I{} sh -c "'$PY' $RUN points --fields {} > '$LOGS/points_{}.log' 2>&1 || { echo 'failed: {}'; exit 255; }"
}
stage_rows() { say "rows"; "$PY" $RUN rows --out "$WORK/rows" > "$LOGS/rows.log" 2>&1; tail -2 "$LOGS/rows.log"; }
stage_report() { say "report"; "$PY" $RUN report --rows "$WORK/rows" --out "$WORK/report" > "$LOGS/report.log" 2>&1; say "written: $WORK/report/report.txt"; }
stage_dryrun() {
  say "dry run on BB.72 (code check only)"
  local D=$WORK/dryrun; mkdir -p "$D"
  SOILGP_DATA=${DRY_DATA:?set DRY_DATA} SOILGP_RUN=$D/run "$PY" $RUN points --fields BB.72 --reps 300 301 > "$LOGS/dry_points.log" 2>&1
  SOILGP_DATA=$DRY_DATA SOILGP_RUN=$D/run "$PY" $RUN rows --fields BB.72 --reps 300 301 --out "$D/rows" --workers 2 > "$LOGS/dry_rows.log" 2>&1
  SOILGP_DATA=$DRY_DATA SOILGP_RUN=$D/run "$PY" $RUN report --fields BB.72 --rows "$D/rows" --out "$D/report" > "$LOGS/dry_report.log" 2>&1
  say "dry run finished"
}

stages=("$@"); [ "${stages[0]:-}" = all ] && stages=(fetch export data points rows report)
for s in "${stages[@]}"; do "stage_$s"; done
