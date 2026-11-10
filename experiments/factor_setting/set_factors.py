"""Variant selection on the development splits and calibration on the first set.

  python -m experiments.factor_setting.set_factors select --rows DIR --out DIR
      reads DIR/rows_dev_*.csv -> selection.json and a score table
  python -m experiments.factor_setting.set_factors calibrate --rows DIR --selection selection.json --out DIR
      reads DIR/rows_first_*.csv -> calibration.json

Selection: within each comparator, on the blocks where all its variants are scored, each variant's score is its CRPS ratio against TabPFN
with independent errors (geometric mean over fields, averaged over field and zone means), at factor 1. The lowest score wins; within 0.5 %
the variant listed first in config.GROUPS (the simplest) wins.
Calibration: for each selected variant (and the sample mean), the smallest factor on 0.50-5.00 (step 0.05) whose mean of pooled field- and
zone-mean 80 % coverage on all first-set rows (LimeSoDa and all Oxford/Meuse rows, as stored) reaches 0.80. The same
rule applied to J1p must return 1.10, which checks the implementation. Most favourable version for the CRPS endpoint: calibrated or
uncalibrated, whichever has the lower CRPS on the first set's LimeSoDa rows (ratio of the two versions of the same variant).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.core import config as F
from experiments.core import scoring


SKIP_KAPPA_CHECK = False                                                      # --skip-kappa-check: small tests on a subset only


def read_rows(rows_dir, set_name):
    files = sorted(Path(rows_dir).glob(f"rows_{set_name}_*.csv"))
    assert files, f"no rows_{set_name}_*.csv in {rows_dir}"
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def select(rows_dir, out):
    r = scoring.score(read_rows(rows_dir, "dev")); table, selection = [], {}
    for group, variants in F.GROUPS.items():
        if group == "Sample mean":
            selection[group] = "DB"; continue
        common = scoring.common_blocks(r, variants + ["TAB"])
        scores = {v: scoring.score_vs_tab(common, v) for v in variants}
        best = min(scores.values())
        chosen = next(v for v in variants if scores[v] <= best * (1 + F.TIE))
        selection[group] = chosen
        n_blocks = int(scoring.kind_rows(common, "field").query("variant == 'TAB'").shape[0])
        for v in variants:
            table.append(dict(group=group, variant=v, score=scores[v], selected=v == chosen, common_field_blocks=n_blocks,
                              failed_share=float(r[r.variant == v].failed.mean())))
    pd.DataFrame(table).to_csv(Path(out) / "selection_scores.csv", index=False)
    (Path(out) / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    print(pd.DataFrame(table).to_string(index=False)); print(json.dumps(selection, indent=1))


def factor_for(r, variant):
    """Smallest factor on the grid whose mean of pooled field and zone coverage reaches 0.80 (rows where the variant is scored)."""
    s = r[(r.variant == variant) & ~r.failed]
    sd = np.sqrt(np.clip(s.V.to_numpy(), 0, None)); df = s.df.to_numpy(); err = np.abs(s.obs.to_numpy() - s.M.to_numpy())
    q = np.where(np.isfinite(df), scoring.tdist.ppf(0.9, np.where(np.isfinite(df), df, 2.0)), F.Z80); field = (s.block == "field").to_numpy()
    cov = {}
    for f in F.FACTOR_GRID:
        c = err <= q * f * sd
        cov[float(f)] = float((c[field].mean() + c[~field].mean()) / 2)
    chosen = next((f for f, v in cov.items() if v >= F.COVERAGE_TARGET), float(F.FACTOR_GRID[-1]))
    return chosen, cov


def calibrate(rows_dir, selection_file, out):
    r = read_rows(rows_dir, "first"); selection = json.loads(Path(selection_file).read_text())
    k_ours, cov_ours = factor_for(r, "J1p")
    check = {"J1p_factor": k_ours, "J1p_coverage_at_1.05": cov_ours[1.05], "J1p_coverage_at_1.10": cov_ours[1.1]}
    if not SKIP_KAPPA_CHECK:
        assert abs(k_ours - F.KAPPA_OURS) < 1e-9, f"the rule does not reproduce kappa = 1.10 on the first set: {check}"
    lim = r[r.src == "limesoda"]; result = {"check": check, "comparators": {}}
    for group, v in selection.items():
        f, cov = factor_for(r, v)
        # calibrated vs uncalibrated version of the same variant: CRPS ratio (ratio of mean CRPS per task, geometric mean over fields,
        # averaged over field and zone means). Comparing each with TabPFN would be degenerate when the variant is TabPFN itself.
        s = lim[lim.variant == v]; both = pd.concat([s, s.assign(variant=v + "__cal")], ignore_index=True)
        scored = scoring.score(both, {v + "__cal": f})
        ratio = float(np.exp(np.mean([np.log(scoring.contrast(scored, v + "__cal", v, kind)["ratio"]) for kind in ("field", "zone")])))
        result["comparators"][group] = {"variant": v, "factor": f, "coverage_at_factor": cov[f], "coverage_at_1": cov[1.0],
                                        "crps_ratio_calibrated_to_uncalibrated": ratio,
                                        "crps_version": "calibrated" if ratio < 1 else "uncalibrated"}
    (Path(out) / "calibration.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["select", "calibrate"])
    ap.add_argument("--rows", required=True); ap.add_argument("--selection"); ap.add_argument("--out", required=True)
    ap.add_argument("--skip-kappa-check", action="store_true", help="small tests on a subset only"); a = ap.parse_args()
    global SKIP_KAPPA_CHECK
    SKIP_KAPPA_CHECK = a.skip_kappa_check
    Path(a.out).mkdir(parents=True, exist_ok=True)
    select(a.rows, a.out) if a.stage == "select" else calibrate(a.rows, a.selection, a.out)


if __name__ == "__main__":
    main()
