#!/usr/bin/env bash
# E12 of docs/exploratory/plan_E9-E13.md: the study with seeds 1-4 of TabPFN and of the fold split (predictions, area rows, report),
# seed by seed.
#   WORK=<the study's work folder> PY=<python> SOILGP_RLIB=<R library> JOBS=8 bash experiments/exploratory/run_seeds.sh
# Writes $WORK/run_seed<s>, $WORK/rows_seed<s> and $WORK/report_seed<s>/report.txt; the rows and report of one seed run while the next
# seed's predictions are made. Seed 0 is the study itself ($WORK/report/report.txt).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${WORK:?set WORK to the work folder of the study}; PY=${PY:-python}; JOBS=${JOBS:-8}
case "$PY" in */*) PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY");; esac     # every module runs from the repository root
W=$(cd "$W" && pwd) || exit 1
SEEDS="-m experiments.exploratory.seed_study"
export SOILGP_DATA=$W/data PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOGS=$W/seedlogs; mkdir -p "$LOGS"
cd "$REPO"
fields=$(SOILGP_RUN=$W/run "$PY" -c "from experiments.study import run as s; print(' '.join(s.evaluated()))" 2>/dev/null)
rows_report() {
  local s=$1
  SOILGP_RUN=$W/run_seed$s "$PY" $SEEDS rows "$s" "$W/rows_seed$s" 6 > "$LOGS/rows_$s.log" 2>&1
  SOILGP_RUN=$W/run_seed$s "$PY" -m experiments.study.run report --rows "$W/rows_seed$s" --out "$W/report_seed$s" > "$LOGS/report_$s.log" 2>&1
  echo "$(date '+%F %T') seed $s: report written" >> "$LOGS/driver.log"
}
for s in 1 2 3 4; do
  echo "$(date '+%F %T') seed $s: predictions" >> "$LOGS/driver.log"
  printf '%s\n' $fields | xargs -P "$JOBS" -I{} sh -c "SOILGP_RUN=$W/run_seed$s '$PY' $SEEDS points $s {} > '$LOGS/points_${s}_{}.log' 2>&1 || echo 'failed: seed $s {}' >> '$LOGS/driver.log'"
  echo "$(date '+%F %T') seed $s: predictions done" >> "$LOGS/driver.log"
  rows_report "$s" &
done
wait
echo "$(date '+%F %T') all seeds done" >> "$LOGS/driver.log"
