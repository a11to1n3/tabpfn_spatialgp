"""Exploratory sensitivity analyses and checks of TabPFN-GP's assumptions (the paper's settings table).

Reads the saved point outputs (block_cases.load_case); CPU only, no TabPFN refits, no R. Each variant changes one fixed choice of the chosen
block model (J1p, scored with kappa = 1.10 unless the report says otherwise):
  M12, M52  Matern smoothness 1/2 and 5/2, each with its own population prior, derived from the development fits as the chosen prior was
  X3        the prior derived without SP.231, SL.125 and UL.120 (the report evaluates it without these fields and NSW.52)
  EB        a normal prior on the offset instead of the flat one: beta ~ N(beta0, psi) on the standardised scale, beta0 = -mean(r) / sd(r)
            (no mean error on the original scale), psi learned by moments on the development cases behind the prior (mean square of the
            standardised GLS offset minus its mean variance 1 / (1' K^-1 1)), floored at zero, where the offset is known and the offset term
            vanishes; beta and the offset term of the block variance are their posterior versions, and the GP's mean correction applies at
            every n. The priors stage also gives these moments by n on all development splits.
  GC        the grid of spatial shares centred on the posterior mode, logit share = logit(share at the mode) + k ln 3, k = -2..2, within
            the shares that the variance bounds allow (the chosen grid is centred on 0.5)
  TQ        the GP part of the block variance multiplied by (t_{0.9, n-1} / z_{0.9})^2
  J10, LOO  residuals cross-fitted with J = min(10, n) folds or one fold per revealed location (folds.py, GPU; final set, 20 fields)
Zones (final set): the chosen model, the offset-only model (O1) and the sample mean (DB) on other zones: k-means of the coordinates with 2
and 8 clusters (all fields), and k-means of the principal components of the standardised covariates (90 % of their variance) with 4
clusters (c4) and with one cluster per 20 ha of the field's convex hull, between 2 and 8 (ca), on the 14 fields whose covariates can be
mapped densely; as for the chosen zones, a zone is evaluated where it holds at least 3 unrevealed locations.
Checks:
  folds     shared errors of the fold models (assumption A1): the between-fold mean square of the cross-fitted residuals divided by their
            sample variance, which has expectation 1 over the random fold assignment when the folds share no error, and estimates the
            factor by which such errors would inflate the variance of the offset (n >= 10, so that every fold holds two residuals)
  trunc     how often TabPFN's excess variance max(0, v_T - v_bar) is truncated at zero, and the coverage of zone means by its share
  grid      how often the chosen grid puts its highest weight on the spatial share 0.1
  pooled    assumption A3 (the same range and share distribution at every n): flat-prior fits pooled over the 10 splits with n = 20, and
            over those with n = 50, of the first set on the 14 development fields, against the prior from n >= 50

  python -m experiments.sensitivity.sensitivity priors --out DIR [--workers 16]      -> DIR/sensitivity_priors.json, DIR/dev_fits.csv
  python -m experiments.sensitivity.sensitivity rows --set first|final --priors DIR/sensitivity_priors.json --out DIR [--oof FOLDS_DIR] [--workers 16]
  python -m experiments.sensitivity.sensitivity pooled --out DIR [--workers 16]      -> DIR/pooled_fits.csv
  python -m experiments.sensitivity.sensitivity report --rows DIR --final-rows DIR --first-rows DIR --models-first-rows DIR --out DIR
--rows holds the outputs of the other stages; --final-rows and --first-rows the stored rows of the final and first sets (read-only);
--models-first-rows the first-set rows of pipeline/further_models.py (the offset-only model's factor); --oof the output folder of folds.py (its
5-fold residuals, if present, are compared with the stored ones). Writes DIR/report_sensitivity.txt.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import json                                                                   # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from scipy.linalg import cho_factor, cho_solve                                # noqa: E402
from scipy.optimize import minimize                                           # noqa: E402
from scipy.spatial.distance import cdist                                      # noqa: E402
from scipy.special import expit, logit                                        # noqa: E402
from scipy.stats import t as tdist                                            # noqa: E402
from sklearn.cluster import KMeans                                            # noqa: E402
from sklearn.model_selection import KFold                                     # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.sensitivity import models_report as RR                     # noqa: E402  (factor_rule)
from experiments.development.derive_priors import block_prior                 # noqa: E402
from experiments.sensitivity.models_report import KEY                         # noqa: E402
from experiments.factor_setting.set_factors import read_rows                  # noqa: E402
from experiments.paths import CONFIGS                                         # noqa: E402
from experiments.core.block_model import PRIORS                               # noqa: E402
from experiments.core.point_model import FLAT                                 # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent, matern32   # noqa: E402
from tabpfn_spatialgp.preprocess import spectral_mask_from_columns            # noqa: E402

C = F.C
POP = PRIORS["POP"]
GP_VARIANTS = ("J1p", "M12", "M52", "X3", "EB", "GC", "TQ")
FOLD_VARIANTS = (("10", "J10"), ("loo", "LOO"))
LABELS = {"J1p": "chosen model", "M12": "Matern-1/2", "M52": "Matern-5/2", "X3": "prior without SP.231, SL.125, UL.120",
          "EB": "normal prior on the offset", "GC": "share grid centred on the mode", "TQ": "t quantile for the GP part",
          "J10": "10 folds", "LOO": "leave-one-out residuals"}
EXCLUDED_PRIOR = ("SP.231", "SL.125", "UL.120")
EXCLUDED_EVAL = EXCLUDED_PRIOR + ("NSW.52",)
ZONE_SETS = ("k2", "k8", "c4", "ca")
HA_PER_ZONE = 20.0
OFFSET_BINS = [3, 7, 12, 17, 25, 40, 62, 87, 1000]                           # sample sizes of the development splits, for the offset moments
CAL = json.loads((CONFIGS / "fixed" / "calibration.json").read_text())["comparators"]
COV_F = {CAL[g]["variant"]: CAL[g]["factor"] for g in CAL}
CRPS_F = {CAL[g]["variant"]: (CAL[g]["factor"] if CAL[g]["crps_version"] == "calibrated" else 1.0) for g in CAL}
MODEL_COMPARATORS = [CAL[g]["variant"] for g in ("TabPFN", "TabPFN-KpR", "Block RK", "WH")]


# ------------------------------------------------------------------------------------------------------------------------- the GP
def matern12(D, ell):
    """Matern-1/2 (exponential) correlation: exp(-d / ell)."""
    return np.exp(-D / ell)


def matern52(D, ell):
    """Matern-5/2 correlation: (1 + sqrt(5) d / ell + 5 d^2 / (3 ell^2)) exp(-sqrt(5) d / ell)."""
    r = np.sqrt(5.0) * D / ell
    return (1 + r + r ** 2 / 3) * np.exp(-r)


KERNELS = {"M32": matern32, "M12": matern12, "M52": matern52}


class KField(ResidualField):
    """ResidualField with another correlation function and, optionally, a normal prior on the offset. With matern32 and no offset prior
    it computes exactly what ResidualField computes (the report checks it against the stored rows)."""

    def __init__(self, prior, extent, kern=matern32):
        super().__init__(prior, extent); self.kern = kern

    def _nlp(self, th, ret=False):                                           # ResidualField._nlp with self.kern
        p = self._unpack(th); n = len(self.y)
        K = p["s1"] * self.kern(self.D, p["l1"]) + np.diag(np.full(n, p["tau"])) + S_.JITTER * np.eye(n)
        try:
            Fc = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        H = self.H; KiH = cho_solve(Fc, H); A = H.T @ KiH; beta = np.linalg.solve(A, KiH.T @ self.y)
        res = self.y - H @ beta; al = cho_solve(Fc, res)
        nll = 0.5 * (res @ al) + np.log(np.diag(Fc[0])).sum() + 0.5 * np.linalg.slogdet(A)[1]
        share = p["s1"] / (p["s1"] + p["tau"])
        dl = (np.log(p["l1"] / self.ext) - self.prior["mu_r"]) / self.prior["s_r"]
        ds = (np.log(share / (1 - share + 1e-12)) - self.prior["mu_s"]) / self.prior["s_s"]
        nll += 0.5 * (dl ** 2 + ds ** 2)
        return (nll, Fc, beta, al) if ret else nll

    def moments(self, coords_q, blocks, offset_prior=None):
        """ResidualField.block_moments with self.kern. offset_prior = (beta0, var) on the standardised scale: the GLS mean is replaced by
        its posterior mean under beta ~ N(beta0, var) and 1' K^-1 1 by 1' K^-1 1 + 1 / var in the offset term (Rasmussen and Williams,
        2006, Section 2.7); var <= 0 means a known offset beta0, without offset term."""
        Sq = np.asarray(coords_q, float); Dq = cdist(Sq, self.S); Dqq = cdist(Sq, Sq); per_theta = []
        for th in self.thetas:
            p = self._unpack(th); _, Fc, beta, al = self._nlp(th, ret=True)
            kq = p["s1"] * self.kern(Dq, p["l1"]); Kqq = p["s1"] * self.kern(Dqq, p["l1"])
            Kikq = cho_solve(Fc, kq.T); C0 = Kqq - kq @ Kikq
            A = float((self.H.T @ cho_solve(Fc, self.H)).ravel()[0]); u = 1.0 - (self.H.T @ Kikq).ravel()
            b = float(np.ravel(beta)[0])
            if offset_prior is None:
                C = C0 + np.outer(u, u) / A
            else:
                b0, vb = offset_prior
                if vb > 0:
                    A = A + 1.0 / vb; b = (float((self.H.T @ cho_solve(Fc, self.y)).ravel()[0]) + b0 / vb) / A; C = C0 + np.outer(u, u) / A
                else:                                                         # known offset: no offset term
                    b, C = b0, C0
                al = cho_solve(Fc, self.y - b)
            mu = b + kq @ al
            per_theta.append((self._vbar_theta(Fc, p), {k: (float(mu[i].mean()), float(C[np.ix_(i, i)].mean())) for k, i in blocks.items()}))
        vbar = float(np.dot(self.w, [t[0] for t in per_theta])); out = {}
        for k in blocks:
            m_t = np.array([t[1][k][0] for t in per_theta]); v_t = np.array([t[1][k][1] for t in per_theta]); mb = float(np.dot(self.w, m_t))
            out[k] = (self.ym + self.ys * mb, self.ys ** 2 * (float(np.dot(self.w, v_t + m_t ** 2)) - mb ** 2))
        return out, self.ys ** 2 * vbar

    def centre_shares(self, min_range):
        """Replace the averaging grid by one with the same ranges and the spatial shares centred on the posterior mode (see GC)."""
        base = self.map_theta; tot = np.exp(base[0]) + np.exp(base[2]); lo_v, hi_v = S_.VARIANCE_BOUNDS
        lo, hi = max(lo_v / tot, 1 - hi_v / tot), min(hi_v / tot, 1 - lo_v / tot)
        centre = logit(np.exp(base[0]) / tot)
        shares = expit(np.clip(centre + np.log(3.0) * np.arange(-2, 3), logit(lo), logit(hi)))
        rb = (np.log(min_range), np.log(self.ext) + self.prior["mu_r"] + S_.RANGE_UPPER_LOG_OFFSET)
        grid, lp = [], []
        for dl in S_.GRID_LOG_RANGE_OFFSETS:
            for share in shares:
                th = base.copy(); th[0] = np.log(share * tot); th[2] = np.log((1 - share) * tot); th[1] = np.clip(base[1] + dl, *rb)
                grid.append(th); lp.append(-self._nlp(th))
        w = np.exp(np.array(lp) - max(lp)); self.thetas, self.w = grid, w / w.sum()
        return self


def share_of(th):
    return float(np.exp(th[0]) / (np.exp(th[0]) + np.exp(th[2])))


# ------------------------------------------------------------------------------------------------------------------------- priors
def dev_fits(field):
    """Every development split: the chosen GP's offset at its posterior mode (standardised GLS offset delta and its variance 1/A) and the
    standardised mean residual; and, for the cases behind the prior (n >= 50; as derive_priors.development_fits), flat-prior fits with each
    correlation function (range and share at the grid point with the highest weight)."""
    out = []
    for f, data, design, rep, tr in block_cases.iter_splits("dev", [field], 1):
        S = np.asarray(data["coordinates"], float); ext = field_extent(S); floor = block_range_floor(S[tr])
        for target in C.TARGETS:
            case = block_cases.load_case("dev", f, data, design, rep, tr, target)
            if case is None or case["m_oof"] is None:
                continue
            r = case["y"][tr] - case["m_oof"]
            row = dict(field=f, design=design, rep=rep, target=target, n=len(tr), rbar_s=float(r.mean() / r.std(ddof=1)))
            for name, kern in (KERNELS.items() if len(tr) >= 50 else ()):
                try:
                    g = KField(FLAT, ext, kern).fit(S[tr], r, floor); p = g._unpack(g.thetas[int(np.argmax(g.w))])
                    row[f"range_rel_{name}"] = float(p["l1"] / ext); row[f"share_{name}"] = float(p["s1"] / (p["s1"] + p["tau"]))
                except Exception as e:                                        # counted in the report, never replaced
                    row[f"err_{name}"] = repr(e)[:150]
            try:
                g = KField(POP, ext).fit(S[tr], r, floor); _, Fc, beta, _ = g._nlp(g.map_theta, ret=True)
                row["delta"] = float(np.ravel(beta)[0]) + g.ym / g.ys                  # GLS mean of r in units of sd(r)
                row["inv_A"] = 1.0 / float((g.H.T @ cho_solve(Fc, g.H)).ravel()[0])  # its variance given theta
            except Exception as e:
                row["err_offset"] = repr(e)[:150]
            out.append(row)
    return out


def stage_priors(out, workers):
    with Pool(min(workers, len(C.DEV_FIELDS))) as pool:
        fits = pd.DataFrame([r for res in pool.map(dev_fits, C.DEV_FIELDS) for r in res])
    fits.to_csv(out / "dev_fits.csv", index=False, float_format="%.17g")
    ref = pd.read_csv(CONFIGS / "prior_fits.csv").set_index(["field", "design", "rep", "target"])
    big = fits[fits.n >= 50]; got = big.set_index(["field", "design", "rep", "target"]); both = got.index.intersection(ref.index)
    diff = max(float((got.loc[both, f"{c}_M32"] - ref.loc[both, c]).abs().max()) for c in ("range_rel", "share"))
    pri = {}
    for name in KERNELS:
        ok = big.dropna(subset=[f"range_rel_{name}", f"share_{name}"])
        pri[name] = block_prior(ok.rename(columns={f"range_rel_{name}": "range_rel", f"share_{name}": "share"}))
        pri[f"{name}_fits"] = int(len(ok))
    kept = ref.reset_index(); kept = kept[~kept.field.isin(EXCLUDED_PRIOR)]
    pri["X3"] = block_prior(kept); pri["X3_fits"] = int(len(kept))
    o = big.dropna(subset=["delta", "inv_A"])
    pri["offset"] = {"psi": float(np.mean(o.delta ** 2) - np.mean(o.inv_A)), "mean_delta": float(o.delta.mean()),
                     "mean_delta2": float(np.mean(o.delta ** 2)), "mean_inv_A": float(o.inv_A.mean()), "cases": int(len(o))}
    a = fits.dropna(subset=["delta", "inv_A"]).assign(bin=lambda d: pd.cut(d.n, OFFSET_BINS))
    pri["offset_by_n"] = [{"bin": str(b), "cases": int(len(s)), "n_median": float(s.n.median()), "mean_delta2": float(np.mean(s.delta ** 2)),
                           "mean_inv_A": float(s.inv_A.mean()), "mean_rbar_s2": float(np.mean(s.rbar_s ** 2)), "mean_inv_n": float(np.mean(1 / s.n))}
                          for b, s in a.groupby("bin", observed=True)]
    pri["check"] = {"M32_matched": int(len(both)), "M32_cases": int(len(ref)), "M32_max_abs_diff": diff, "M32_prior_equals_POP": pri["M32"] == POP}
    (out / "sensitivity_priors.json").write_text(json.dumps(pri, indent=1) + "\n"); print(json.dumps(pri, indent=1))


# ------------------------------------------------------------------------------------------------------------------------- zones
def covariate_scores(data):
    """Principal components (90 % of the variance) of the field's standardised covariates, spectra excluded, at all its locations."""
    X = np.asarray(data["X"], float)[:, ~spectral_mask_from_columns(data["columns"])]
    X = X[:, np.nanstd(X, 0) > 0]; assert np.all(np.isfinite(X)), "non-finite covariates"
    Z = (X - X.mean(0)) / X.std(0); U, s, _ = np.linalg.svd(Z, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.90) + 1)
    return U[:, :k] * s[:k]


def zone_labels(field, data):
    """Cluster labels of every location of the field for each zone set (see the module docstring)."""
    S = np.asarray(data["coordinates"], float); km = lambda X, k: KMeans(k, random_state=0, n_init=10).fit_predict(X)   # noqa: E731
    labels = {"k2": km(S, 2), "k8": km(S, 8)}
    if field in F.DENSE_FIELDS:
        Z = covariate_scores(data); ka = int(min(8, max(2, np.floor(field_extent(S) ** 2 / 1e4 / HA_PER_ZONE + 0.5))))
        labels["c4"], labels["ca"] = km(Z, 4), km(Z, ka)
    return labels


def extra_blocks(labels, tr, te):
    blocks, revealed, zset = {}, {}, {}
    for name, lab in labels.items():
        for j in range(int(lab.max()) + 1):
            idx = np.flatnonzero(lab[te] == j)
            if len(idx) >= S_.ZONE_MIN_POINTS:
                b = f"{name}z{j}"; blocks[b], revealed[b], zset[b] = idx, tr[lab[tr] == j], name
    return blocks, revealed, zset


# ------------------------------------------------------------------------------------------------------------------------- rows
def fold_ratio(r):
    """Between-fold mean square of the residuals over their sample variance, for the fold assignment of core/point_model.py."""
    n = len(r)
    if n < 10:
        return np.nan
    folds = [b for _, b in KFold(min(S_.OOF_FOLDS, n), shuffle=True, random_state=S_.RANDOM_STATE).split(np.zeros(n))]
    msb = sum(len(b) * (r[b].mean() - r.mean()) ** 2 for b in folds) / (len(folds) - 1)
    return float(msb / r.var(ddof=1))


def alternative_oof(oof_dir, folds, case):
    """Cross-fitted means of folds.py for this case, or None."""
    if not oof_dir:
        return None
    path = block_cases.point_file(Path(oof_dir) / folds, "ours", case["design"], case["field"], case["target"], case["rep"])
    if not path.exists():
        return None
    d = np.load(path); assert np.array_equal(d["train_rows"], case["tr"]), path
    return d["m_oof"]


def case_rows(case, P, labels, oof_dir=None):
    S, tr, te, y = case["S"], case["tr"], case["te"], case["y"]; n = len(tr); yq = y[te]; mT, sT = case["mT"], case["sT"]
    gate = n >= F.GATE_N; L = field_extent(S); floor = block_range_floor(S[tr])
    blocks, revealed = block_cases.blocks_of(case)
    zb, zrev, zset = extra_blocks(labels, tr, te) if labels else ({}, {}, {})
    allb, allrev = {**blocks, **zb}, {**revealed, **zrev}
    key = dict(set=case["set"], src="external" if case["field"] in C.EXTERNAL else "limesoda", field=case["field"], design=case["design"],
               rep=case["rep"], target=case["target"], n=n, m=len(te))
    rows = []

    def add(variant, b, M, V, df=np.inf, err="", **extra):
        i = allb[b]; failed = bool(err) or not (np.isfinite(M) and np.isfinite(V) and V > 1e-18)
        rows.append(dict(key, block=b, zones=zset.get(b, "k4"), mB=len(i), nB=len(allrev[b]), variant=variant, M=float(M), V=float(V),
                         df=float(df), obs=float(yq[i].mean()), failed=failed, err=err or ("non-finite mean or variance <= 1e-18" if failed else ""),
                         **extra))

    def assemble(variant, moments, vbar, these, correct, gp_mult=1.0, trunc=False):
        excess = np.clip(sT ** 2 - vbar, 0, None)
        for b in these:
            i = allb[b]; mb, vb = moments[b]; extra = {"trunc": float(np.mean(sT[i] ** 2 < vbar))} if trunc else {}
            add(variant, b, float(mT[i].mean()) + (mb if correct else 0.0), gp_mult * vb + float(np.sum(excess[i])) / len(i) ** 2, **extra)

    def failed_all(variants, these, e):
        for v in variants:
            for b in these:
                add(v, b, np.nan, np.nan, err=repr(e)[:150])

    diag = dict(key)
    if case["m_oof"] is None:
        failed_all(("J1p", "O1") + GP_VARIANTS[1:], allb, ValueError("no cross-fitted residuals")); return rows, diag
    r = y[tr] - case["m_oof"]; rbar, s2 = float(r.mean()), float(r.var(ddof=1))
    diag.update(fold_ratio=fold_ratio(r), s_r=float(np.sqrt(s2)))
    try:                                                                      # the chosen model, on every block
        g = KField(POP, L).fit(S[tr], r, floor); mom, vbar = g.moments(S[te], allb)
        assemble("J1p", mom, vbar, allb, gate, trunc=True)
        assemble("TQ", mom, vbar, blocks, gate, gp_mult=(tdist.ppf(0.9, n - 1) / F.Z80) ** 2)
        top = g.thetas[int(np.argmax(g.w))]
        diag.update(vbar=vbar, share_mode=share_of(g.map_theta), share_top=share_of(top), range_rel_mode=float(np.exp(g.map_theta[1]) / L))
        try:
            eb, vbar_eb = g.moments(S[te], blocks, offset_prior=(-g.ym / g.ys, max(P["offset"]["psi"], 0.0)))
            assemble("EB", eb, vbar_eb, blocks, True)
        except Exception as e:
            failed_all(("EB",), blocks, e)
        try:
            gc, vbar_gc = g.centre_shares(floor).moments(S[te], blocks)
            assemble("GC", gc, vbar_gc, blocks, gate)
        except Exception as e:
            failed_all(("GC",), blocks, e)
    except Exception as e:                                                    # counted and reported, never replaced
        failed_all(("J1p",), allb, e); failed_all(("TQ", "EB", "GC"), blocks, e)
    for v, prior, kern in (("M12", P["M12"], matern12), ("M52", P["M52"], matern52), ("X3", P["X3"], matern32)):
        try:
            gv = KField(prior, L, kern).fit(S[tr], r, floor); mv, vv = gv.moments(S[te], blocks)
            assemble(v, mv, vv, blocks, gate)
        except Exception as e:
            failed_all((v,), blocks, e)
    for folds, v in FOLD_VARIANTS:
        m_alt = alternative_oof(oof_dir, folds, case)
        if m_alt is None:
            continue
        try:
            ga = KField(POP, L).fit(S[tr], y[tr] - m_alt, floor); ma, va = ga.moments(S[te], blocks)
            assemble(v, ma, va, blocks, gate)
        except Exception as e:
            failed_all((v,), blocks, e)
    m5 = alternative_oof(oof_dir, "5", case)
    if m5 is not None:
        diag["j5_max_abs_diff"] = float(np.max(np.abs(m5 - case["m_oof"])))
    for b, i in allb.items():                                                 # offset only and the sample mean, on every block
        m = len(i); mTB = float(mT[i].mean())
        add("O1", b, mTB + (rbar if gate else 0.0), s2 / n + float(np.sum(sT[i] ** 2)) / m ** 2)
        yr = y[allrev[b]]
        if len(yr) >= 2:
            add("DB", b, float(yr.mean()), float(np.var(yr, ddof=1)) * (1 / len(yr) + 1 / m), df=len(yr) - 1)
    return rows, diag


def field_rows(job):
    set_name, field, priors_file, out_dir, oof_dir = job
    assert block_cases.FINAL_POINTS == F.POINTS["final"], "the final set's point outputs are redirected"
    P = json.loads(Path(priors_file).read_text()); t0 = time.monotonic(); rows, diags, missing = [], [], 0; labels = None
    for f, data, design, rep, tr in block_cases.iter_splits(set_name, [field]):
        if labels is None:
            labels = zone_labels(f, data) if (set_name == "final" and f in C.LIMESODA) else {}
        for target in F.targets(f, set_name):
            if not np.all(np.isfinite(data["Y"][:, C.TARGETS.index(target)])):
                continue
            case = block_cases.load_case(set_name, f, data, design, rep, tr, target)
            if case is None:
                missing += 1; continue
            rr, d = case_rows(case, P, labels, oof_dir); rows += rr; diags.append(d)
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"sens_{set_name}_{field}.csv", index=False)
    pd.DataFrame(diags).to_csv(Path(out_dir) / f"diag_{set_name}_{field}.csv", index=False)
    return field, len(rows), missing, time.monotonic() - t0


# ------------------------------------------------------------------------------------------------------------------------- pooled
def pooled_fit(field, target, design):
    """Flat-prior fit with (sigma^2, range, tau^2) shared by the 10 splits of a design of the first set, each split's residuals standardised
    on their own: range and share at the grid point with the highest weight of the chosen averaging grid around the posterior mode, as for
    the fits behind the prior (development/derive_priors.py), and at the posterior mode itself."""
    parts, floors, ext = [], [], None
    for f, data, d, rep, tr in block_cases.iter_splits("first", [field]):
        if d != design:
            continue
        case = block_cases.load_case("first", f, data, d, rep, tr, target)
        if case is None or case["m_oof"] is None:
            continue
        S = case["S"]; ext = field_extent(S); r = case["y"][tr] - case["m_oof"]
        g = ResidualField(FLAT, ext); g.S = S[tr]; g.ym, g.ys = float(np.mean(r)), float(np.std(r)) or 1.0
        g.y = (r - g.ym) / g.ys; g.D = cdist(g.S, g.S); g.H = np.ones((len(r), 1)); parts.append(g); floors.append(block_range_floor(S[tr]))
    if not parts:
        return None
    nlp = lambda th: sum(g._nlp(th) for g in parts)                          # noqa: E731
    mu_l = np.log(ext) + FLAT["mu_r"]; lv = (np.log(S_.VARIANCE_BOUNDS[0]), np.log(S_.VARIANCE_BOUNDS[1]))
    bounds = [lv, (np.log(max(floors)), mu_l + S_.RANGE_UPPER_LOG_OFFSET), lv]; best = None
    for off in S_.MAP_START_LOG_OFFSETS:
        x0 = np.array([np.log(S_.MAP_START_SIGMA2), np.clip(mu_l + off, *bounds[1]), np.log(S_.MAP_START_TAU2)])
        res = minimize(nlp, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": S_.MAP_MAXITER})
        if best is None or res.fun < best.fun:
            best = res
    grid, lp = [], []
    for dl in S_.GRID_LOG_RANGE_OFFSETS:                                       # the chosen grid (ResidualField.fit) around the pooled mode
        for share in S_.GRID_SPATIAL_SHARES:
            th = best.x.copy(); tot = np.exp(best.x[0]) + np.exp(best.x[2])
            th[0] = np.log(share * tot); th[2] = np.log((1 - share) * tot); th[1] = np.clip(best.x[1] + dl, *bounds[1])
            grid.append(th); lp.append(-nlp(th))
    top = grid[int(np.argmax(lp))]
    return dict(field=field, target=target, design=design, splits=len(parts), range_rel=float(np.exp(top[1]) / ext), share=share_of(top),
                range_rel_mode=float(np.exp(best.x[1]) / ext), share_mode=share_of(best.x))


def pooled_field(field):
    return [p for target in C.TARGETS for design in ("n20", "n50") if (p := pooled_fit(field, target, design)) is not None]


# ------------------------------------------------------------------------------------------------------------------------- report
def report(rows_dir, final_rows, first_rows, rev_first, out):
    lines = []

    def say(s=""):
        lines.append(s); print(s, flush=True)

    rd = Path(rows_dir); P = json.loads((rd / "sensitivity_priors.json").read_text())
    rd_csv = lambda pat: pd.concat([pd.read_csv(p, dtype={"err": str}) for p in sorted(rd.glob(pat))], ignore_index=True)   # noqa: E731
    first, final = rd_csv("sens_first_*.csv"), rd_csv("sens_final_*.csv"); diag = rd_csv("diag_final_*.csv")
    say("=== Exploratory sensitivity analyses and checks of TabPFN-GP's assumptions (stored outputs; kappa = 1.10 unless stated) ===")
    ck = P["check"]
    say(f"\n-- priors: Matern-3/2 fits recomputed for {ck['M32_matched']} of the {ck['M32_cases']} development cases, largest difference "
        f"{ck['M32_max_abs_diff']:.1e}; the derived prior equals the chosen one: {ck['M32_prior_equals_POP']} --")
    for v, k in (("M12", "M12"), ("M52", "M52"), ("X3", "X3")):
        p = P[k]; say(f"  {v:4s} {LABELS[v]:38s} mu_r {p['mu_r']:.2f}, s_r {p['s_r']:.2f}, mu_s {p['mu_s']:.2f}, s_s {p['s_s']:.2f} "
                      f"({P[k + '_fits']} fits)")
    o = P["offset"]
    say(f"  EB   offset prior: psi = mean(delta^2) - mean(1/A) = {o['mean_delta2']:.4f} - {o['mean_inv_A']:.4f} = {o['psi']:.4f} "
        f"(mean delta {o['mean_delta']:.4f}; {o['cases']} cases); used: {max(o['psi'], 0.0):.4f}"
        f"{' (known offset, no offset term)' if o['psi'] <= 0 else ''}")
    say("  offset moments on every development split, by n: mean delta^2 against mean 1/A (GLS), and mean (rbar / s_r)^2 against mean 1/n")
    for q in P["offset_by_n"]:
        say(f"    n in {q['bin']:11s} (median {q['n_median']:5.1f}, {q['cases']:3d} cases): {q['mean_delta2']:.4f} vs {q['mean_inv_A']:.4f}; "
            f"{q['mean_rbar_s2']:.4f} vs {q['mean_inv_n']:.4f}")
    for set_name, mine, reg_dir in (("first", first, first_rows), ("final", final, final_rows)):
        ref = read_rows(reg_dir, set_name); a = mine[(mine.variant == "J1p") & (mine.zones == "k4")].set_index(KEY)
        b = ref[ref.variant == "J1p"].set_index(KEY); both = a.index.intersection(b.index)
        rel = max(float((np.abs(a.loc[both, c] - b.loc[both, c]) / np.abs(b.loc[both, c]).clip(lower=1e-12)).max()) for c in ("M", "V"))
        say(f"  check, {set_name} set: J1p against the stored rows: {len(both)} of {len(a)} blocks matched, largest relative "
            f"difference {rel:.1e}")
    k4 = lambda d: d[d.zones == "k4"]                                         # noqa: E731

    # ---------------------------------------------------------------- 1. factors by the rule
    say("\n-- 1. Factor by the factor rule on the first set (all its rows, as for kappa) --")
    fac = {v: RR.factor_rule(k4(first), v, F.FACTOR_GRID) for v in GP_VARIANTS}
    say("  " + "; ".join(f"{v} {fac[v]:.2f}" for v in GP_VARIANTS))
    say("  the rule's criterion, the mean of pooled field and zone coverage on the first set, at factors 1.00, 1.05, 1.10 and 1.15:")
    for v in GP_VARIANTS:
        s = k4(first); s = s[(s.variant == v) & ~s.failed]; err = np.abs(s.obs - s.M).to_numpy(); sd = np.sqrt(s.V.clip(lower=0)).to_numpy()
        fld = (s.block == "field").to_numpy(); crit = lambda f: ((err <= F.Z80 * f * sd)[fld].mean() + (err <= F.Z80 * f * sd)[~fld].mean()) / 2   # noqa: E731
        say(f"    {v:4s} " + ", ".join(f"{crit(f):.4f}" for f in (1.00, 1.05, 1.10, 1.15)))
    folds_present = [v for _, v in FOLD_VARIANTS if v in set(final.variant)]
    if "j5_max_abs_diff" in diag:
        d5 = diag.j5_max_abs_diff.dropna()
        say(f"  check: 5-fold residuals recomputed by folds.py for {len(d5)} cases, largest absolute difference from the stored ones "
            f"{d5.max():.1e}")
    fl = k4(final)[k4(final).src == "limesoda"]
    variants = list(GP_VARIANTS) + folds_present
    sc = scoring.score(fl, {v: F.KAPPA_OURS for v in variants}); own = scoring.score(fl, fac)
    common = scoring.common_blocks(sc, variants); keys = common.set_index(KEY).index.unique()
    inc = lambda d: d[d.set_index(KEY).index.isin(keys)]                      # noqa: E731
    cs, co = inc(sc), inc(own)
    nf = int(((cs.variant == "J1p") & (cs.block == "field")).sum()); nz = int(((cs.variant == "J1p") & (cs.block != "field")).sum())
    groups = (("n <= 20", lambda d: d[d.n <= 20]), ("n >= 30", lambda d: d[d.n >= 30]))

    # ---------------------------------------------------------------- 2. model choices at kappa = 1.10
    say(f"\n-- 2. Final splits, 20 fields, blocks where every variant is scored ({nf} field, {nz} zone blocks), kappa = 1.10: coverage "
        f"(field | zone) and CRPS ratio variant / chosen model [95 % interval] --")
    for v in variants:
        if v == "X3":
            continue
        parts = []
        for lab, sel in groups:
            cov = [scoring.pooled_coverage(sel(cs), v, k, boot=False)[0] for k in ("field", "zone")]
            txt = f"{lab}: {cov[0]:.3f} | {cov[1]:.3f}"
            if v != "J1p":
                cr = [scoring.contrast(sel(cs), v, "J1p", k) for k in ("field", "zone")]
                txt += "; CRPS " + " | ".join(RR_ci(c) for c in cr)
            parts.append(txt)
        say(f"  {v:4s} {LABELS[v]:34s} " + "  ||  ".join(parts))
    keep16 = [f for f in C.LIMESODA if f not in EXCLUDED_EVAL]
    say(f"  without SP.231, NSW.52, SL.125 and UL.120 ({len(keep16)} fields):")
    for v in ("J1p", "X3"):
        parts = []
        for lab, sel in groups:
            s = sel(cs[cs.field.isin(keep16)]); cov = [scoring.pooled_coverage(s, v, k, boot=False)[0] for k in ("field", "zone")]
            txt = f"{lab}: {cov[0]:.3f} | {cov[1]:.3f}"
            if v != "J1p":
                txt += "; CRPS " + " | ".join(RR_ci(scoring.contrast(s, v, "J1p", k)) for k in ("field", "zone"))
            parts.append(txt)
        say(f"  {v:4s} {LABELS[v]:34s} " + "  ||  ".join(parts))
    reg = read_rows(final_rows, "final"); reg = reg[(reg.src == "limesoda") & reg.field.isin(keep16)]
    x3 = fl[(fl.variant == "X3") & fl.field.isin(keep16)].assign(variant="X3")
    both = pd.concat([reg[reg.variant.isin(MODEL_COMPARATORS + ["DB", "J1pK"])], x3[[c for c in x3.columns if c in reg.columns]]],
                     ignore_index=True)
    bs = scoring.score(both, {**CRPS_F, "X3": F.KAPPA_OURS})
    cm = scoring.common_blocks(bs, MODEL_COMPARATORS + ["J1pK", "X3"])
    say("  primary CRPS contrasts on these fields, X3 / comparator and the chosen model / comparator (field | zone; comparators on the "
        "common set, the sample mean on all blocks where both are scored):")
    for comp in MODEL_COMPARATORS + ["DB"]:
        src = bs if comp == "DB" else cm
        say(f"    {comp:13s} X3 " + " | ".join(RR_ci(scoring.contrast(src, "X3", comp, k)) for k in ("field", "zone")) +
            "   chosen " + " | ".join(RR_ci(scoring.contrast(src, "J1pK", comp, k)) for k in ("field", "zone")))

    # ---------------------------------------------------------------- 3. variants with their own factor
    say("\n-- 3. The same blocks, each variant with its own factor from 1.: pooled coverage, largest distance from 0.80 over n, and "
        "CRPS ratio variant / chosen model (field | zone) --")
    for v in GP_VARIANTS:
        cov = [scoring.pooled_coverage(co, v, k, boot=False)[0] for k in ("field", "zone")]
        dist = [max(abs(scoring.pooled_coverage(co[co.n == n], v, k, boot=False)[0] - 0.8) for n in sorted(co.n.unique()))
                for k in ("field", "zone")]
        txt = f"  {v:4s} factor {fac[v]:.2f}: coverage {cov[0]:.3f} | {cov[1]:.3f}, largest distance {dist[0]:.3f} | {dist[1]:.3f}"
        if v != "J1p":
            txt += "; CRPS " + " | ".join(RR_ci(scoring.contrast(co, v, "J1p", k)) for k in ("field", "zone"))
        say(txt)
    say("  coverage by n with the own factor (field | zone):")
    say("    n    " + "  ".join(f"{v:>13s}" for v in GP_VARIANTS))
    for n in sorted(co.n.unique()):
        s = co[co.n == n]
        say(f"    {int(n):3d}  " + "  ".join(f"{scoring.pooled_coverage(s, v, 'field', boot=False)[0]:.3f} | "
                                         f"{scoring.pooled_coverage(s, v, 'zone', boot=False)[0]:.3f}" for v in GP_VARIANTS))

    # ---------------------------------------------------------------- 4. zones
    f_o1, c_o1 = own_factor_o1(read_rows_rev(rev_first))
    f_db = COV_F["DB"]
    say(f"\n-- 4. Other zones, final splits (zone means only): TabPFN-GP (1.10), the offset-only model ({f_o1:.2f}, CRPS version "
        f"{'with' if c_o1 != 1.0 else 'without'} it) and the sample mean ({f_db:.2f} for coverage, CRPS without) --")
    zl = final[(final.src == "limesoda") & (final.block != "field") & final.variant.isin(["J1p", "O1", "DB"])]
    zc = scoring.score(zl, {"J1p": F.KAPPA_OURS, "O1": f_o1, "DB": f_db}); zk = scoring.score(zl, {"J1p": F.KAPPA_OURS, "O1": c_o1})
    dense = list(F.DENSE_FIELDS)
    for name, fields in (("k4", None), ("k2", None), ("k8", None), ("k4", dense), ("c4", dense), ("ca", dense)):
        sel = lambda d: d[(d.zones == name) & (d.field.isin(fields) if fields else True)]   # noqa: E731
        a, b = sel(zc), sel(zk); nzb = int(((a.variant == "J1p")).sum())
        tag = f"{name}{' (14 fields)' if fields else ' (20 fields)'}"
        parts = []
        for lab, g in groups:
            ga, gb = g(a), g(b)
            cov = {v: scoring.pooled_coverage(ga, v, "zone", boot=False)[0] for v in ("J1p", "O1", "DB")}
            emp = {v: float(ga[(ga.variant == v) & (ga.nB == 0)].cov80.mean()) for v in ("J1p", "O1")}
            n_emp = int(((ga.variant == "J1p") & (ga.nB == 0)).sum())
            r_o1, r_db = scoring.contrast(gb, "J1p", "O1", "zone"), scoring.contrast(gb, "J1p", "DB", "zone")
            parts.append(f"{lab}: coverage GP {cov['J1p']:.3f}, O1 {cov['O1']:.3f}, DB {cov['DB']:.3f}; no samples ({n_emp} zones) GP "
                         f"{emp['J1p']:.3f}, O1 {emp['O1']:.3f}; CRPS GP/O1 {RR_ci(r_o1)}, GP/DB {RR_ci(r_db)}")
        say(f"  {tag:16s} {nzb:6d} zones  " + "\n" + " " * 18 + ("\n" + " " * 18).join(parts))

    # ---------------------------------------------------------------- 5. checks
    say("\n-- 5. Checks of the assumptions (final splits, 20 fields) --")
    dl = diag[diag.src == "limesoda"]
    say("  folds (A1): between-fold mean square / residual variance (expectation 1 without shared fold errors), by n: mean [95 % "
        "interval over fields], median")
    rng = np.random.default_rng(F.SEED)
    for n in sorted(dl.n.unique()):
        s = dl[(dl.n == n) & dl.fold_ratio.notna()]
        if s.empty:
            continue
        g = s.groupby("field").fold_ratio.agg(["sum", "count"]); idx = rng.integers(0, len(g), (F.B, len(g)))
        bs_ = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1); lo, hi = np.percentile(bs_, [2.5, 97.5])
        say(f"    n = {int(n):3d}: {s.fold_ratio.mean():.3f} [{lo:.3f}, {hi:.3f}], median {s.fold_ratio.median():.3f} ({len(s)} cases)")
    jr = final[(final.src == "limesoda") & (final.variant == "J1p") & (final.zones == "k4")]
    js = scoring.score(jr, {"J1p": F.KAPPA_OURS})
    say("  truncation of max(0, v_T - v_bar): share of unrevealed locations truncated (field blocks), share of zone blocks with any "
        "truncation, by n")
    for n in sorted(js.n.unique()):
        s = js[js.n == n]
        say(f"    n = {int(n):3d}: locations {s[s.block == 'field'].trunc.mean():.3f}; zone blocks {(s[s.block != 'field'].trunc > 0).mean():.3f}")
    z30 = js[(js.n >= 30) & (js.block != "field")]
    for lab, m in (("none truncated", z30.trunc == 0), ("up to half", (z30.trunc > 0) & (z30.trunc <= 0.5)), ("more than half", z30.trunc > 0.5)):
        say(f"    n >= 30, zone coverage with {lab:15s}: {z30[m].cov80.mean():.3f} ({int(m.sum())} zone blocks)")
    for lab, s in (("n <= 20", dl[dl.n <= 20]), ("n >= 30", dl[dl.n >= 30])):
        say(f"  share grid (chosen): highest weight on the share 0.1 in {np.mean(np.isclose(s.share_top, 0.1)):.3f} of the cases with {lab}; "
            f"posterior-mode share below 0.1 in {np.mean(s.share_mode < 0.1):.3f}")
    say("  posterior-mode spatial share and range across cases, by n: median and interquartile range of logit(share) and of log(range / L); "
        f"the prior's interquartile ranges are {1.349 * POP['s_s']:.2f} and {1.349 * POP['s_r']:.2f}")
    for n in sorted(dl.n.unique()):
        s = dl[dl.n == n]; ls_, lr_ = logit(s.share_mode), np.log(s.range_rel_mode)
        q = lambda x: (np.median(x), np.subtract(*np.percentile(x, [75, 25])))   # noqa: E731
        say(f"    n = {int(n):3d}: logit(share) {q(ls_)[0]:+.2f} (IQR {q(ls_)[1]:.2f}); log(range / L) {q(lr_)[0]:+.2f} (IQR {q(lr_)[1]:.2f})")
    pf = pd.read_csv(rd / "pooled_fits.csv"); lr, ls = np.log(pf.range_rel), logit(pf.share)
    say(f"  pooled fits (A3), first set, 14 development fields; prior (n >= 50): log(range / L) mean {POP['mu_r']:.2f}, sd {POP['s_r']:.2f}; "
        f"logit(share) mean {POP['mu_s']:.2f}, sd {POP['s_s']:.2f}")
    for d in ("n20", "n50"):
        m = pf.design == d
        say(f"    {d}: {int(m.sum())} field-properties, log(range / L) mean {lr[m].mean():.2f}, sd {lr[m].std():.2f}; logit(share) mean "
            f"{ls[m].mean():.2f}, sd {ls[m].std():.2f}")
    w = pf.pivot_table(index=["field", "target"], columns="design", values=["range_rel", "share"]).dropna()
    say(f"    same field and property at n = 20 and 50 ({len(w)}): median |difference| of log(range / L) "
        f"{np.median(np.abs(np.log(w['range_rel']['n20']) - np.log(w['range_rel']['n50']))):.2f}, of logit(share) "
        f"{np.median(np.abs(logit(w['share']['n20']) - logit(w['share']['n50']))):.2f}")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "report_sensitivity.txt").write_text("\n".join(lines) + "\n")


def RR_ci(c):
    return "-" if c is None else f"{c['ratio']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]"


def own_factor_o1(first):
    """The offset-only model's factor and CRPS version, by the factor rule on the first-set rows."""
    fac = RR.factor_rule(first, "O1", F.FACTOR_GRID)
    s = first[(first.src == "limesoda") & (first.variant == "O1")]
    both = scoring.score(pd.concat([s, s.assign(variant="O1__cal")], ignore_index=True), {"O1__cal": fac})
    rcal = float(np.exp(np.mean([np.log(scoring.contrast(both, "O1__cal", "O1", k)["ratio"]) for k in ("field", "zone")])))
    return fac, (fac if rcal < 1 else 1.0)


def read_rows_rev(d):
    return pd.concat([pd.read_csv(p, dtype={"err": str}) for p in sorted(Path(d).glob("models_first_*.csv"))], ignore_index=True)


# ------------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["priors", "rows", "pooled", "report"])
    ap.add_argument("--set", choices=["first", "final"]); ap.add_argument("--priors"); ap.add_argument("--out", required=True)
    ap.add_argument("--oof")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--rows"); ap.add_argument("--final-rows")
    ap.add_argument("--first-rows"); ap.add_argument("--models-first-rows"); ap.add_argument("--fields", nargs="*", help="tests only")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.stage == "priors":
        stage_priors(out, a.workers)
    elif a.stage == "rows":
        fields = a.fields or sorted({f for group, _, _ in F.SETS[a.set] for f in group})
        with Pool(min(a.workers, len(fields))) as pool:
            for field, n, missing, sec in pool.imap_unordered(field_rows, [(a.set, f, a.priors, str(out), a.oof) for f in fields]):
                print(f"{a.set} {field}: {n} rows, {missing} cases without point outputs, {sec:.0f} s", flush=True)
    elif a.stage == "pooled":
        with Pool(min(a.workers, len(C.DEV_FIELDS))) as pool:
            fits = pd.DataFrame([p for res in pool.map(pooled_field, C.DEV_FIELDS) for p in res])
        fits.to_csv(out / "pooled_fits.csv", index=False, float_format="%.17g"); print(fits.to_string())
    else:
        report(a.rows, a.final_rows, a.first_rows, a.models_first_rows, a.out)


if __name__ == "__main__":
    main()
