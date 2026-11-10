"""Point outputs of one split (GPU; R for the kriging of TabPFN-KpR): our TabPFN with cross-fitted residuals, and TabPFN-KpR.
Called by pipeline/points.py and study/run.py, which write the files.

our_model -> the 'ours' file: TabPFN v2.0.9 with the coordinates as inputs, 4 members, no target transform, and
    none_mean, none_crps, none_quantiles, none_var   TabPFN's predictive distribution at the test rows (quantiles at config.QUANTILES)
    bb_m_oof, bb_v_oof                              cross-fitted TabPFN means and variances at the training rows (J = min(5, n) folds)
    m_ins                                           in-sample TabPFN means at the training rows
    Gbma_mean, Gbma_crps, Gbma_quantiles            the package's point predictive (TabPFN + point GP); not part
                                                    of the paper's results. Below 8 training points it equals TabPFN alone (flagged).
kpr -> 'KpR' / 'KpR4': kriging prior regression (Schmidinger et al., 2026), TabPFN on the sensing covariates plus the leave-one-out kriging
    mean and variance at the training rows and the kriging at the test rows (kriging.R). KpR uses TabPFN's default settings, as
    Schmidinger et al. (2026) did; KpR4 uses ours. If kriging fails, both fall back to TabPFN (default settings) on the sensing covariates.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # TabPFN's output depends on the thread count:
    os.environ[_v] = "1"                                                      # the study used one thread

import json                                                                   # noqa: E402

import numpy as np                                                            # noqa: E402
from sklearn.model_selection import KFold                                     # noqa: E402

from experiments.core import datasets as C                                    # noqa: E402
from experiments.paths import CONFIGS                                         # noqa: E402
from experiments.core.common import S_, bar_output, bars, covariates, default_tabpfn, kriging, tabpfn  # noqa: E402
from tabpfn_spatialgp.bars import GaussianMixture, quantile_levels            # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, field_extent, point_range_floor   # noqa: E402

PRIORS = json.loads((CONFIGS / "priors.json").read_text())
FLAT = {"mu_r": float(np.log(0.15)), "s_r": np.inf, "mu_s": 0.0, "s_s": np.inf}   # no prior term (the fits behind the block prior)


def point_prior(field, target):
    """Prior of the point GP: from pilot fits on BB.250, BB.72, G.150 and MG.112; each of these four gets the prior of the other three."""
    return PRIORS["point"][target].get(field, PRIORS["point"][target]["all"])


def our_model(field, data, design, rep, tr, te, target, backbone="ours", variants=("Gbma",)):
    """Our TabPFN with cross-fitted residuals and the point GP; returns the arrays saved in the 'ours' file."""
    assert backbone == "ours" and tuple(variants) == ("Gbma",), "only the evaluated model is kept"
    make = lambda: tabpfn(n_estimators=4, target_transform=False)                       # noqa: E731  (coordinates as inputs)
    xa, xb = covariates(data, tr, te, with_coords=True)
    S = np.asarray(data["coordinates"], float); y = data["Y"][:, C.TARGETS.index(target)]; yt, yq = y[tr], y[te]
    final = make().fit(xa, yt); bq = bars(final, xb); mean_T, var_T = bq.moments()
    out = {f"none_{k}": v for k, v in bar_output(bq, yq).items()}; out["none_var"] = var_T
    info = {"n_train": int(len(tr)), "backbone": backbone, "fails": {}}
    m_oof, v_oof = np.empty(len(tr)), np.empty(len(tr))
    for a, b in KFold(min(S_.OOF_FOLDS, len(tr)), shuffle=True, random_state=S_.RANDOM_STATE).split(xa):
        if np.ptp(yt[a]) == 0:                                   # constant training fold (rounded lab values): predict the constant
            m_oof[b], v_oof[b] = yt[a][0], max(float(np.var(yt)), 1e-12); continue
        m_oof[b], v_oof[b] = bars(make().fit(xa[a], yt[a]), xa[b]).moments()
    out.update(bb_m_oof=m_oof, bb_v_oof=v_oof, m_ins=bars(final, xa).moments()[0])
    ext = field_extent(S); prior = point_prior(field, target); floor = point_range_floor(S[tr], prior, ext)
    info.update(prior=prior, min_len=floor, gp_fallback_backbone_only=bool(len(tr) < 8))
    centers = bq.quantiles(quantile_levels(S_.N_MIXTURE))
    for v in variants:
        if len(tr) < 8:                                           # too few points for any spatial model: TabPFN alone (flagged)
            out.update({f"{v}_{k}": out[f"none_{k}"] for k in ("mean", "crps", "quantiles")}); continue
        r = yt - m_oof
        try:
            g = ResidualField(prior, ext).fit(S[tr], r, floor)
            mu, var_g, vbar = g.predict_points(S[te])
            mix = GaussianMixture(centers + mu[:, None], np.sqrt(np.clip(var_g - vbar, 0, None)))
            o = bar_output(mix, yq); o["mean"] = mean_T + mu                   # exact TabPFN mean + GP mean, as in the study
            out.update({f"{v}_{k}": x for k, x in o.items()})
        except Exception as e:                                    # a failed fit falls back to TabPFN alone (recorded)
            info["fails"][v] = repr(e)[:200]
            out.update({f"{v}_{k}": out[f"none_{k}"] for k in ("mean", "crps", "quantiles")})
    return out, info


def kpr(data, tr, te, target, which):
    """KpR and/or KpR4 on one split. Returns {model: arrays}, kriging_failed."""
    xa, xb = covariates(data, tr, te, with_coords=False); y = data["Y"][:, C.TARGETS.index(target)]; yt, yq = y[tr], y[te]
    try:
        k = kriging("kpr", data["coordinates"], tr, te, yt)
        fa, fb = k["kpr_training"], k["kpr_query"]
        assert np.array_equal(fa.row, tr) and np.array_equal(fb.row, te)
        xa, xb = np.column_stack([xa, fa[["mean", "variance"]]]), np.column_stack([xb, fb[["mean", "variance"]]]); failed = False
    except Exception as e:                                        # kriging failed: TabPFN (default settings) on the sensing covariates
        failed = repr(e)[:300]
    out = {}
    for m in which:
        reg = tabpfn() if (m == "KpR4" and not failed) else default_tabpfn()
        out[m] = bar_output(bars(reg.fit(xa, yt), xb), yq)
    return out, failed
