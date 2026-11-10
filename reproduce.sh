#!/usr/bin/env bash
# Reproduces the study of the paper from the public data, in six stages (README.md describes each one).
#   bash reproduce.sh all                  every stage in order
#   bash reproduce.sh data dev prior ...   the named stages, in the order given
# Stages: data dev prior factor_setting study compare
# Settings (environment variables): WORK (default ./work), everything this script writes; PY (default python), the Python of
# environment/requirements.txt; JOBS (default 6), parallel GPU jobs for the point outputs; SOILGP_RLIB, the R library if not the default.
# Point outputs that already exist are skipped, so an interrupted stage is resumed by running it again.
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
WORK=$(mkdir -p "${WORK:-$REPO/work}" && cd "${WORK:-$REPO/work}" && pwd)
PY=${PY:-python}
case "$PY" in */*) PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY");; esac     # every module runs from the repository root
JOBS=${JOBS:-6}
FS=$WORK/factor_setting
LOGS=$WORK/logs
mkdir -p "$LOGS"
export SOILGP_DATA=$WORK/data SOILGP_RUN=$WORK/run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore

LIMESODA_REPO=https://github.com/a11to1n3/LimeSoDa.git
LIMESODA_COMMIT=89ad14a944af2d0cf60d6aeec72b3ba675a59c14          # the LimeSoDa data of the study
SENSRES_URL="https://zenodo.org/records/14161809/files/SensRes_Lithuania.zip?download=1"
SENSRES_MD5=7fe9110bb5bfc3b423f49767c943a8a0                        # as listed on Zenodo (doi:10.5281/zenodo.14161809)

say() { echo "$(date '+%F %T') $*" | tee -a "$LOGS/reproduce.log"; }

# The fields of a set, largest first (the point outputs run one job per field, JOBS at a time).
fields_of() {
  (cd "$REPO" && "$PY" -c "
import numpy as np
from experiments.core import config as F
fs = sorted({f for g, _, _ in F.SETS['$1'] for f in g}, key=lambda f: -len(np.load(F.C.DATA / f'{f}.npz')['Y']))
print(' '.join(fs))")
}

# One point-output job per field: points <set> <log prefix> <command ...>; the command gets --fields <field>.
points() {
  local set=$1 tag=$2; shift 2
  local fields; fields=$(fields_of "$set")
  say "  point outputs ($tag): $(echo $fields | wc -w) fields, $JOBS at a time"
  printf '%s\n' $fields | xargs -P "$JOBS" -I{} sh -c "cd '$REPO' && $* --fields {} > '$LOGS/${tag}_{}.log' 2>&1 || { echo 'failed: {} (see $LOGS/${tag}_{}.log)'; exit 255; }"
}

stage_data() {
  say "1 data: sources and field files"
  mkdir -p "$WORK/sources" && cd "$WORK/sources"
  [ -d LimeSoDa ] || git clone -q "$LIMESODA_REPO" LimeSoDa
  git -C LimeSoDa checkout -q "$LIMESODA_COMMIT"
  [ -f SensRes_Lithuania.zip ] || curl -fsSL -o SensRes_Lithuania.zip "$SENSRES_URL"
  echo "$SENSRES_MD5  SensRes_Lithuania.zip" | md5sum -c --quiet -
  [ -d SensRes_Lithuania ] || unzip -q SensRes_Lithuania.zip
  cd "$REPO"
  "$PY" -m experiments.data.export_data --limesoda "$WORK/sources/LimeSoDa/src" --sensres "$WORK/sources/SensRes_Lithuania" | tail -1
  "$PY" -m experiments.data.checks fingerprint | tee "$LOGS/1_fingerprint.log" | tail -1
}

stage_dev() {
  say "2 dev: point outputs of the development splits (GPU)"
  points dev points_dev "$PY" -m experiments.pipeline.points --set dev --min-test 1
}

stage_prior() {
  say "3 prior: the fits behind the prior, and the prior"
  cd "$REPO"
  "$PY" -m experiments.development.derive_priors fits --out "$WORK/prior_fits.csv" | tee "$LOGS/3_prior_fits.log"
  "$PY" -m experiments.development.derive_priors block --fits "$WORK/prior_fits.csv" | tee "$WORK/prior_check.txt"
}

stage_factor_setting() {
  say "4 factor_setting: the comparators' variants and the factors (first set: GPU)"
  cd "$REPO"
  "$PY" -m experiments.pipeline.block_rows shapes --out "$FS" > "$LOGS/4_shapes.log" 2>&1
  "$PY" -m experiments.pipeline.block_rows rows --set dev --shapes "$FS/shapes.json" --out "$FS" > "$LOGS/4_rows_dev.log" 2>&1
  "$PY" -m experiments.factor_setting.set_factors select --rows "$FS" --out "$FS" | tee "$LOGS/4_select.log" | tail -2
  points first points_first "$PY" -m experiments.pipeline.points --set first
  "$PY" -m experiments.pipeline.block_rows rows --set first --shapes "$FS/shapes.json" --out "$FS" > "$LOGS/4_rows_first.log" 2>&1
  "$PY" -m experiments.factor_setting.set_factors calibrate --rows "$FS" --selection "$FS/selection.json" --out "$FS" | tee "$LOGS/4_calibrate.log" | tail -2
  "$PY" -m experiments.factor_setting.analysis --rows "$FS" --set first --selection "$FS/selection.json" --calibration "$FS/calibration.json" \
    --out "$FS" > "$LOGS/4_dry_run.log" 2>&1
  "$PY" -m experiments.compare factor-setting "$WORK" | tee "$LOGS/4_compare.log"
}

stage_study() {
  say "5 study: the study of docs/PROTOCOL.md on 43 fields (fetch, export, predictions, area rows, report; GPU)"
  WORK="$WORK/study" PY="$PY" JOBS="$JOBS" bash "$REPO/experiments/study/run_study.sh" all
  (cd "$WORK/study/report" && "$PY" "$REPO/paper/fig_coverage_by_n.py" coverage_by_n.csv fig_coverage_by_n > "$LOGS/5_figure.log" 2>&1)
}

stage_compare() {
  say "6 compare: the reports against expected/"
  cd "$REPO"
  "$PY" -m experiments.compare reports "$WORK" | tee "$LOGS/6_compare.log"
}

STAGES=(data dev prior factor_setting study compare)
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 1; }
[ "$1" = all ] && set -- "${STAGES[@]}"
for s in "$@"; do
  declare -F "stage_$s" > /dev/null || { echo "unknown stage: $s (stages: ${STAGES[*]})"; exit 1; }
  "stage_$s"
done
say "done: $*"
