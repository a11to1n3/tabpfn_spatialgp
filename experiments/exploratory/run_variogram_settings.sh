#!/usr/bin/env bash
# E18 of docs/exploratory/plan_E18.md (CPU and R). Stops at the first failed check.
#   W=<study work folder> REPRO=<reproduce.sh work folder, with factor_setting/ and run/points_dev, points_first> PY=<python> \
#     bash experiments/exploratory/run_variogram_settings.sh
# Settings: WORKERS (default 10); SOILGP_RLIB, the R library (gstat, automap) if not the default. E18=$W/e18 receives everything.
set -Eeuo pipefail
: "${W:?}" "${REPRO:?}" "${PY:?}"
R=$(cd "$(dirname "$0")/../.." && pwd)
W=$(cd "$W" && pwd); REPRO=$(cd "$REPRO" && pwd)
case "$PY" in */*) PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY");; esac     # every module runs from the repository root
WORKERS=${WORKERS:-10}
E18=$W/e18; FS=$REPRO/factor_setting
VS="-m experiments.exploratory.variogram_settings"; SF="-m experiments.factor_setting.set_factors"
mkdir -p "$E18"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd "$R"
say() { echo "$(date '+%F %T') $*" | tee -a "$E18/driver.log"; }
fail() { say "STOPPED at line $1"; touch "$E18/FAILED"; }
trap 'fail $LINENO' ERR
rm -f "$E18/FAILED" "$E18/DONE"

stage_env() { SOILGP_DATA=$REPRO/data SOILGP_RUN=$REPRO/run "$@"; }
study_env() { SOILGP_DATA=$W/data SOILGP_RUN=$W/run "$@"; }

say "check 1: the medians"
"$PY" $VS settings "$FS/shape_fits.csv" "$E18" > "$E18/check1.log" 2>&1
tail -1 "$E18/check1.log" | tee -a "$E18/driver.log"

say "check 2: the unchanged settings reproduce the stored rows"
C2=$E18/check2; mkdir -p "$C2"
stage_env "$PY" $VS rows dev "$E18/shapes_ALL.json" "$C2" 3 BB.72 > "$E18/check2_dev.log" 2>&1
stage_env "$PY" $VS rows first "$E18/shapes_ALL.json" "$C2" 3 MWP.36 > "$E18/check2_first.log" 2>&1
study_env "$PY" $VS rows study "$E18/shapes_ALL.json" "$C2" 3 NI_BRA > "$E18/check2_study.log" 2>&1
mkdir -p "$C2/stage" "$C2/studyfiles"
mv -f "$C2"/fix_dev_*.csv "$C2"/fix_first_*.csv "$C2/stage/"
mv -f "$C2"/fix_study_*.csv "$C2/studyfiles/"
stage_env "$PY" $VS compare "$C2/stage" "$FS" > "$E18/check2_cmp.log" 2>&1
study_env "$PY" $VS compare "$C2/studyfiles" "$W/rows" >> "$E18/check2_cmp.log" 2>&1
tail -1 "$E18/check2_cmp.log" | tee -a "$E18/driver.log"

say "check 3: the stored rows give back the study's choices and report"
C3=$E18/check3; mkdir -p "$C3"
stage_env "$PY" $SF select --rows "$FS" --out "$C3" > "$E18/check3_select.log" 2>&1
stage_env "$PY" $SF calibrate --rows "$FS" --selection "$C3/selection.json" --out "$C3" > "$E18/check3_cal.log" 2>&1
"$PY" -c "
import json, sys
fx, c3 = sys.argv[1], sys.argv[2]
assert json.load(open(fx + '/selection.json')) == json.load(open(c3 + '/selection.json')), 'selection differs'
a, b = json.load(open(fx + '/calibration.json'))['comparators'], json.load(open(c3 + '/calibration.json'))['comparators']
for g in a:
    assert a[g]['variant'] == b[g]['variant'] and a[g]['factor'] == b[g]['factor'] and a[g]['crps_version'] == b[g]['crps_version'], (g, a[g], b[g])
print('versions and factors reproduced')
" "$R/configs/fixed" "$C3" | tee -a "$E18/driver.log"
mkdir -p "$C3/report"
study_env "$PY" $VS report "$W/rows" "$R/configs" "$C3/report" > "$E18/check3_report.log" 2>&1
cmp "$C3/report/report.txt" "$W/report/report.txt"
say "check 3 passed: report identical to the study's"

for X in X2 X3; do
  D=$E18/$X; mkdir -p "$D/fix" "$D/rows" "$D/fixed"
  say "$X: rows of the development splits"
  stage_env "$PY" $VS rows dev "$E18/shapes_$X.json" "$D/fix" "$WORKERS" > "$D/rows_dev.log" 2>&1
  say "$X: rows of the factor-setting data"
  stage_env "$PY" $VS rows first "$E18/shapes_$X.json" "$D/fix" "$WORKERS" > "$D/rows_first.log" 2>&1
  say "$X: rows of the 43 fields"
  study_env "$PY" $VS rows study "$E18/shapes_$X.json" "$D/fix" "$WORKERS" > "$D/rows_study.log" 2>&1
  say "$X: merge, version, factor, report"
  "$PY" $VS merge "$D/fix" "$FS" dev "$D/rows" > "$D/merge.log" 2>&1
  "$PY" $VS merge "$D/fix" "$FS" first "$D/rows" >> "$D/merge.log" 2>&1
  "$PY" $VS merge "$D/fix" "$W/rows" study "$D/rows" >> "$D/merge.log" 2>&1
  stage_env "$PY" $SF select --rows "$D/rows" --out "$D/fixed" > "$D/select.log" 2>&1
  stage_env "$PY" $SF calibrate --rows "$D/rows" --selection "$D/fixed/selection.json" --out "$D/fixed" > "$D/calibrate.log" 2>&1
  study_env "$PY" $VS report "$D/rows" "$D" "$D/report" > "$D/report.log" 2>&1
  say "$X: done"
done

say "summary"
study_env "$PY" -m experiments.exploratory.variogram_settings_report "$W/rows" "$E18" "$E18/summary" > "$E18/summary.log" 2>&1
say "all done"
touch "$E18/DONE"
