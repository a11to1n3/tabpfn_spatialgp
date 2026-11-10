"""Derive the block prior of TabPFN-GP (field and zone means) from the development fits, and compare it with configs/priors.json.

The prior comes from GP fits on the development splits: one fit per split and property with n >= 50 on the 14 development fields
(random designs, replicates 15-19; flat prior, field block), with the spatial share and the range relative to the field extent at the grid
point with the highest weight (configs/prior_fits.csv holds the 630 fits). Each prior is the mean and sd (divisor n - 1) of
log(range / extent) and of logit(spatial share), rounded to two decimals:
  'POP'   all 14 fields (= tabpfn_spatialgp.BLOCK_PRIOR)
  'LOFO'  for each development field, the prior from the other 13 (the secondary analysis with a leave-one-field-out prior)
The point priors in priors.json serve only the package's point predictions, which the paper does not evaluate; they come from pilot fits
on BB.250, BB.72, G.150 and MG.112 and are kept as constants.

  python -m experiments.development.derive_priors fits [--fields F ...] [--out CSV]
      recomputes the fits (default: all 14 development fields) from the development point outputs and compares them with prior_fits.csv
  python -m experiments.development.derive_priors block [--fits CSV] [--write]
      the block priors from the fits (default: prior_fits.csv), compared with priors.json; --write replaces priors.json['block']
"""
import argparse
import json

import numpy as np
import pandas as pd

from experiments.core import datasets as C
from experiments.paths import CONFIGS

PRIORS = CONFIGS / "priors.json"
FITS = CONFIGS / "prior_fits.csv"
KEY = ["field", "design", "rep", "target"]


def block_prior(fits):
    lr = np.log(fits.range_rel); ls = np.log(fits.share / (1 - fits.share))
    return {"mu_r": round(float(lr.mean()), 2), "s_r": round(float(lr.std()), 2), "mu_s": round(float(ls.mean()), 2), "s_s": round(float(ls.std()), 2)}


def block_priors(fits):
    return {"POP": block_prior(fits), "LOFO": {f: block_prior(fits[fits.field != f]) for f in C.DEV_FIELDS}}


def development_fits(fields):
    """The fits behind the block prior, recomputed from the development point outputs (the cross-fitted TabPFN means)."""
    from experiments.core import block_cases
    from experiments.core.point_model import FLAT
    from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent
    rows = []
    for field, data, design, rep, tr in block_cases.iter_splits("dev", list(fields), 1):
        if len(tr) < 50:
            continue
        S = np.asarray(data["coordinates"], float); te = np.setdiff1d(np.arange(len(S)), tr); ext = field_extent(S)
        for target in C.TARGETS:
            case = block_cases.load_case("dev", field, data, design, rep, tr, target)
            if case is None or case["m_oof"] is None:
                continue
            try:
                g = ResidualField(FLAT, ext).fit(S[tr], case["y"][tr] - case["m_oof"], block_range_floor(S[tr]))
                g.block_moments(S[te], {"field": np.arange(len(te))})
            except Exception as e:                                            # reported, never replaced
                print(f"  {field} {design} rep {rep} {target}: fit failed ({e!r})"); continue
            p = g._unpack(g.thetas[int(np.argmax(g.w))])
            rows.append(dict(field=field, design=design, rep=rep, target=target, n=len(tr), range_rel=float(p["l1"] / ext),
                             share=float(p["s1"] / (p["s1"] + p["tau"]))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["fits", "block"]); ap.add_argument("--fields", nargs="*")
    ap.add_argument("--out"); ap.add_argument("--fits", default=str(FITS)); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    if a.what == "fits":
        new = development_fits(a.fields or C.DEV_FIELDS)
        if a.out:
            new.to_csv(a.out, index=False, float_format="%.17g")
        ref_all = pd.read_csv(FITS); expected = int(ref_all.field.isin(new.field.unique()).sum()) if len(new) else 0
        ref = ref_all.set_index(KEY); got = new.set_index(KEY) if len(new) else new; both = got.index.intersection(ref.index) if len(new) else []
        diff = max(float((got.loc[both, c] - ref.loc[both, c]).abs().max()) for c in ("range_rel", "share")) if len(both) else float("nan")
        print(f"{len(new)} fits recomputed; {len(both)} of the {expected} fits of these fields in prior_fits.csv matched, "
              f"largest absolute difference {diff:.1e}")
        return
    pri = json.loads(PRIORS.read_text()); new = block_priors(pd.read_csv(a.fits))
    from tabpfn_spatialgp.priors import BLOCK_PRIOR
    print(f"block prior POP from {a.fits}: {new['POP']}; priors.json: {'identical' if new == pri['block'] else 'DIFFERENT'}; "
          f"the package's BLOCK_PRIOR: {'identical' if new['POP'] == BLOCK_PRIOR else 'DIFFERENT'}")
    if a.write:
        pri["block"] = new; PRIORS.write_text(json.dumps(pri, indent=1) + "\n"); print(f"written to {PRIORS}")


if __name__ == "__main__":
    main()
