"""E10 and E11 of docs/exploratory/plan_E9-E13.md: area rows of further models, from the saved point outputs (CPU).

  first set (factor-setting data):  SOILGP_DATA=<repro>/work/data SOILGP_RUN=<repro>/work/run python -m experiments.exploratory.further_models_rows first <out> [workers]
  the 43 study fields:              SOILGP_DATA=<work>/data SOILGP_RUN=<work>/run python -m experiments.exploratory.further_models_rows study <out> [workers]

Models (rows stored with factor 1; V is the variance of the block mean, as in the study's rows):
  GP    TabPFN-GP as in the study (population prior, averaging): a check, must equal the saved J1p rows
  GPx   prior refitted without SL.125 and UL.120 (dummy coordinates)
  GPnp  no cross-field prior (flat), with averaging
  GPna  prior, no averaging (posterior mode only)
  GPnn  neither (REML mode)
  Z     zone-effect model: f(s) = beta + u_zone(s), u ~ N(0, s_u^2) independent across the four k-means zones, REML on the standardised
        held-out errors; combined with TabPFN as TabPFN-GP is
  O1, DE, DEz   shared-error-only model and difference estimators (formulas of designs.block_rows)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from scipy.linalg import cho_factor, cho_solve                                # noqa: E402
from scipy.optimize import minimize                                           # noqa: E402

SET, OUT = sys.argv[1], Path(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
from experiments.study import run as S                                        # noqa: E402

F = S.F
if SET == "study":
    block_cases = S.setup(S.evaluated())
else:
    from experiments.core import block_cases                                  # noqa: E402
from experiments.pipeline import block_rows                                   # noqa: E402
from experiments.core.block_model import PRIORS                               # noqa: E402
from experiments.development.derive_priors import block_prior                 # noqa: E402
from experiments.paths import CONFIGS                                         # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent   # noqa: E402

FITS = pd.read_csv(CONFIGS / "prior_fits.csv")
PRIOR_X = block_prior(FITS[~FITS.field.isin(["SL.125", "UL.120"])])
FLAT = dict(PRIORS["POP"], s_r=1e6, s_s=1e6)                                  # mu_r kept: it only sets the range bound and the starts


class ModeOnly(ResidualField):
    """TabPFN-GP's GP without averaging: the posterior mode alone."""
    def fit(self, coords, residuals, min_range):
        super().fit(coords, residuals, min_range)
        self.thetas, self.w = [self.map_theta], np.array([1.0])
        return self


def gp_block(cls, prior, Sx, tr, te, y, m_oof, blocks):
    g = cls(prior, field_extent(Sx)).fit(Sx[tr], y[tr] - m_oof, block_range_floor(Sx[tr]))
    return g.block_moments(Sx[te], blocks)


def zone_block(lab_tr, lab_te, r, blocks):
    """Zone-effect model on residuals r: block moments in the original units and v_bar (the leave-one-out latent variance)."""
    ym, ys = float(np.mean(r)), float(np.std(r)) or 1.0
    yv = (r - ym) / ys; n = len(yv)
    Zt = (lab_tr[:, None] == lab_tr[None, :]).astype(float); one = np.ones(n)
    lo, hi = np.log(S_.VARIANCE_BOUNDS[0]), np.log(S_.VARIANCE_BOUNDS[1])

    def nll(th, ret=False):
        su, se = np.exp(th)
        K = su * Zt + se * np.eye(n) + S_.JITTER * np.eye(n)
        try:
            Fk = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        Ki1 = cho_solve(Fk, one); A = one @ Ki1; beta = (Ki1 @ yv) / A
        res = yv - beta; al = cho_solve(Fk, res)
        val = 0.5 * res @ al + np.log(np.diag(Fk[0])).sum() + 0.5 * np.log(A)
        return (val, Fk, beta, al, Ki1, A) if ret else val
    best = None
    for su0 in (0.1, 0.3, 0.6):
        res = minimize(nll, np.log([su0, 1 - su0]), method="L-BFGS-B", bounds=[(lo, hi), (lo, hi)], options={"maxiter": S_.MAP_MAXITER})
        if best is None or res.fun < best.fun:
            best = res
    _, Fk, beta, al, Ki1, A = nll(best.x, ret=True); su, se = np.exp(best.x)
    Ki = cho_solve(Fk, np.eye(n)); vbar = float(np.mean(np.clip(1 / np.diag(Ki) - se, 0, None)))
    out = {}
    for b, i in blocks.items():
        lb = lab_te[i]; m = len(i)
        cstar = su * (lb[:, None] == lab_tr[None, :]).astype(float)          # m x n
        c1 = cstar.sum(0) / m                                                  # 1' C_* / m
        mu = beta + c1 @ al
        counts = np.bincount(lb, minlength=S_.ZONE_K).astype(float)
        cbb = su * float((counts ** 2).sum()) / m ** 2                         # 1' C_** 1 / m^2
        var = cbb - c1 @ cho_solve(Fk, c1) + (1 - c1 @ Ki1) ** 2 / A
        out[b] = (ym + ys * mu, ys ** 2 * max(var, 0.0))
    return out, ys ** 2 * vbar


def case_rows(case):
    Sx, tr, te, y = case["S"], case["tr"], case["te"], case["y"]
    n, yq, mT, sT, m_oof = len(tr), y[te], case["mT"], case["sT"], case["m_oof"]
    blocks, revealed = block_cases.blocks_of(case); gate = n >= F.GATE_N
    from sklearn.cluster import KMeans
    labels = KMeans(S_.ZONE_K, random_state=0, n_init=10).fit_predict(Sx)
    key = dict(set=SET, src="external" if case["field"] in F.C.EXTERNAL else "limesoda", field=case["field"], design=case["design"],
               rep=case["rep"], target=case["target"], n=n, m=len(te))
    rows = []

    def add(variant, b, M, V, df=np.inf):
        i = blocks[b]; failed = not (np.isfinite(M) and np.isfinite(V) and V > 1e-18)
        rows.append(dict(key, block=b, mB=len(i), nB=len(revealed[b]), variant=variant, M=float(M), V=float(V), df=float(df),
                         obs=float(yq[i].mean()), failed=failed))

    def combine(variant, moments, vbar):
        ind = np.clip(sT ** 2 - vbar, 0, None)
        for b, i in blocks.items():
            mb, vb = moments[b]
            add(variant, b, float(mT[i].mean()) + (mb if gate else 0.0), float(np.sum(ind[i])) / len(i) ** 2 + vb)

    for variant, cls, prior in (("GP", ResidualField, PRIORS["POP"]), ("GPx", ResidualField, PRIOR_X), ("GPnp", ResidualField, FLAT),
                                ("GPna", ModeOnly, PRIORS["POP"]), ("GPnn", ModeOnly, FLAT)):
        try:
            mom, vbar = gp_block(cls, prior, Sx, tr, te, y, m_oof, blocks); combine(variant, mom, vbar)
        except Exception:
            for b in blocks:
                add(variant, b, np.nan, np.nan)
    try:
        mom, vbar = zone_block(labels[tr], labels[te], y[tr] - m_oof, blocks); combine("Z", mom, vbar)
    except Exception:
        for b in blocks:
            add("Z", b, np.nan, np.nan)
    r = y[tr] - m_oof; pos = {g: k for k, g in enumerate(tr)}
    rbar, s2 = float(r.mean()), float(r.var(ddof=1))
    for b, i in blocks.items():
        m = len(i); mTB = float(mT[i].mean()); vind = float(np.sum(sT[i] ** 2)) / m ** 2
        add("O1", b, mTB + (rbar if gate else 0.0), s2 / n + vind)
        if b == "field":
            add("DE", b, mTB + rbar, s2 * (1 / n + 1 / m), df=n - 1)
        rb = r[[pos[j] for j in revealed[b]]]
        if b != "field" and len(rb) >= 2:
            add("DEz", b, mTB + float(rb.mean()), float(rb.var(ddof=1)) * (1 / len(rb) + 1 / m), df=len(rb) - 1)
    return rows


def field_job(field):
    t0 = time.monotonic(); rows = []
    for key, case in block_rows.cases(SET, field):
        if case is not None and case["m_oof"] is not None:
            rows += case_rows(case)
    pd.DataFrame(rows).to_csv(OUT / f"variants_{SET}_{field}.csv", index=False)
    return field, len(rows), time.monotonic() - t0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fields = S.evaluated() if SET == "study" else sorted({f for group, _, _ in F.SETS["first"] for f in group})
    todo = [f for f in fields if not (OUT / f"variants_{SET}_{f}.csv").exists()]
    print(f"prior without SL.125 and UL.120: {PRIOR_X} (population: {PRIORS['POP']}); {len(todo)} fields", flush=True)
    with Pool(min(WORKERS, len(todo))) as pool:
        for field, k, sec in pool.imap_unordered(field_job, todo):
            print(f"{SET} {field}: {k} rows, {sec:.0f} s", flush=True)


if __name__ == "__main__":
    main()
