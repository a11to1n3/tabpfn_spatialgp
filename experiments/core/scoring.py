"""Scores and aggregation of the area rows.

score(rows, factors) adds crps, cov80 (central 80 % interval covers the observed block mean) and width (length of that interval) after
multiplying each method's predictive standard deviation by its factor. Normal predictives: closed-form CRPS (Gneiting et al., 2005);
location-scale t (the sample mean): closed-form CRPS (Jordan et al., 2019), finite only for df > 1.
The contrasts reuse the aggregation of aggregate.py: ratio of mean CRPS per task, mean of log-ratios over a field's
properties, geometric mean over fields, percentile bootstrap over fields (10,000, seed 0); Oxford/Meuse by resampling replicates.
"""
import numpy as np
from scipy.special import beta as Bfun
from scipy.stats import norm, t as tdist, wilcoxon

from experiments.core import config as F
from experiments.core.aggregate import boot_fields, external_contrast, pair, proper_by_field


def crps_normal(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_t(y, mu, sd, nu):
    z = (y - mu) / sd
    return sd * (z * (2 * tdist.cdf(z, nu) - 1) + 2 * tdist.pdf(z, nu) * (nu + z ** 2) / (nu - 1)
                 - 2 * np.sqrt(nu) / (nu - 1) * Bfun(0.5, nu - 0.5) / Bfun(0.5, nu / 2) ** 2)


def score(rows, factors=None):
    """Copy of rows with crps, cov80 and width; factors: {variant: factor} (default 1). Failed rows get NaN scores."""
    r = rows.copy(); f = r.variant.map(factors or {}).fillna(1.0).to_numpy()
    sd = np.sqrt(np.clip(r.V.to_numpy(), 0, None)) * f; df = r.df.to_numpy(); mu, y = r.M.to_numpy(), r.obs.to_numpy()
    gauss = ~np.isfinite(df); q = np.where(gauss, F.Z80, tdist.ppf(0.9, np.where(gauss, 2.0, df)))
    crps = np.full(len(r), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):                    # failed rows (NaN or zero variance) are set to NaN below
        crps[gauss] = crps_normal(y[gauss], mu[gauss], sd[gauss])
        tt = (~gauss) & (df > 1)
        crps[tt] = crps_t(y[tt], mu[tt], sd[tt], df[tt])
    r["crps"], r["cov80"], r["width"] = crps, (np.abs(y - mu) <= q * sd).astype(float), 2 * q * sd
    bad = r.failed.to_numpy()
    r.loc[bad, ["crps", "cov80", "width"]] = np.nan
    return r


def kind_rows(r, kind):
    return r[(r.block == "field") == (kind == "field")]


def contrast(r, a, b, kind, col="crps", fields=None):
    """a vs b for one kind of block, on the blocks where both are scored: pooled ratio, bootstrap CI, field values, fields better."""
    s = kind_rows(r, kind)
    if fields is not None:
        s = s[s.field.isin(fields)]
    t = pair(s, a, b, col)
    if t.empty:
        return None
    Lf = proper_by_field(t); est, ci = boot_fields(Lf)
    return {"ratio": est, "ci": ci, "Lf": Lf, "better": int((Lf < 0).sum()), "fields": len(Lf)}


def wilcoxon_less(Lf):
    """One-sided exact Wilcoxon signed-rank test over the field values (H1: a has the lower CRPS), zero differences dropped."""
    x = np.asarray(Lf, float)
    return float(wilcoxon(x, alternative="less", zero_method="wilcox", method="exact").pvalue)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, j in enumerate(order):
        run = max(run, (len(p) - k) * p[j]); adj[j] = min(1.0, run)
    return adj


def pooled_coverage(r, variant, kind, fields=None, boot=True, seed=F.SEED):
    """Pooled 80 % coverage over blocks, with a percentile bootstrap over fields (blocks pooled within the resampled fields)."""
    s = kind_rows(r, kind); s = s[(s.variant == variant) & s.cov80.notna()]
    if fields is not None:
        s = s[s.field.isin(fields)]
    if s.empty:
        return float("nan"), (float("nan"), float("nan"))
    g = s.groupby("field").cov80.agg(["sum", "count"]); est = float(g["sum"].sum() / g["count"].sum())
    if not boot:
        return est, None
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(g), (F.B, len(g)))
    bs = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return est, tuple(np.percentile(bs, [2.5, 97.5]))


def field_mean_coverage(r, variant, kind):
    s = kind_rows(r, kind); s = s[(s.variant == variant) & s.cov80.notna()]
    return float(s.groupby("field").cov80.mean().mean())


def score_vs_tab(r, variant, kinds=("field", "zone")):
    """Selection score: CRPS ratio against TabPFN with independent errors, geometric mean over fields, averaged over field
    and zone means (mean of the two log-ratios)."""
    logs = []
    for kind in kinds:
        c = contrast(r, variant, "TAB", kind)
        logs.append(np.log(c["ratio"]) if c else np.nan)
    return float(np.exp(np.mean(logs)))


def common_blocks(r, variants):
    """Rows restricted to the blocks where every variant in `variants` is scored."""
    key = ["field", "design", "rep", "target", "block"]
    ok = r[r.variant.isin(variants) & r.crps.notna()].groupby(key).variant.nunique()
    keep = ok[ok == len(set(variants))].index
    return r.set_index(key).loc[lambda d: d.index.isin(keep)].reset_index()


def external(r, a, b, kind="field"):
    return external_contrast(r, a, b, kind)
