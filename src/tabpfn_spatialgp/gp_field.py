"""Residual GP: a spatial Gaussian process for TabPFN's out-of-fold errors.

Model (on residuals standardised to mean 0, sd 1; all outputs are transformed back):

    r(s) = beta + g(s) + eps,   g ~ GP(0, sigma^2 k_l),   eps ~ iid N(0, tau^2),   k_l = Matern-3/2 with range l.

* beta: generalised least squares (GLS); theta = (sigma^2, l, tau^2) scored by the restricted log-likelihood plus the log prior.
* Fit: posterior mode (MAP) by L-BFGS-B from three starting ranges, then averaging over a 7 x 5 grid of (range, spatial share) around the
  mode with the total variance sigma^2 + tau^2 held fixed; grid weights proportional to likelihood x prior.
* predict_points: latent mean and variance at query locations, plus v_bar (see below).
* block_moments: mean and variance of the latent field averaged over blocks of query locations (the joint posterior), plus v_bar.

v_bar is the weighted mean, over the grid, of max(0, 1/[K^-1]_ii - tau^2): the leave-one-out latent variance at the training points for a
known beta. TabPFN's own variance already contains this much residual uncertainty, so callers subtract it to avoid counting it twice.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.distance import cdist

from . import settings as S_


def matern32(D, ell):
    """Matern-3/2 correlation at distances D for range ell: (1 + sqrt(3) d / ell) exp(-sqrt(3) d / ell)."""
    r = np.sqrt(3.0) * D / ell
    return (1 + r) * np.exp(-r)


def field_extent(coords):
    """Field extent E = sqrt(area of the convex hull of the sample locations), in coordinate units (metres).
    The study used all sample locations of the field (training and prediction); falls back to the larger side of the bounding box if the
    hull is degenerate (collinear points)."""
    c = np.asarray(coords, float)
    if c.ndim != 2 or c.shape[1] != 2 or not np.all(np.isfinite(c)):
        raise ValueError("coordinates must be a finite (n, 2) array")
    try:
        return float(np.sqrt(ConvexHull(c).volume))
    except QhullError:
        return float(max(np.ptp(c[:, 0]), np.ptp(c[:, 1]), 1.0))


def median_nn_distance(coords):
    """Median distance from each training location to its nearest other training location."""
    c = np.asarray(coords, float)
    return float(np.median(cKDTree(c).query(c, k=2)[0][:, 1]))


def point_range_floor(train_coords, prior, extent):
    """Lower bound of the range for the point model: half of min(median nearest-neighbour distance, prior median range)."""
    return 0.5 * min(median_nn_distance(train_coords), float(np.exp(prior["mu_r"]) * extent))


def block_range_floor(train_coords):
    """Lower bound of the range for the block model: half the median nearest-neighbour distance."""
    return 0.5 * median_nn_distance(train_coords)


class ResidualField:
    """Residual GP with a prior on (log range / extent, logit spatial share) and hyperparameter averaging.

    Parameters are stored as theta = log[sigma^2, range, tau^2]."""

    def __init__(self, prior, extent):
        self.prior, self.ext = dict(prior), float(extent)

    @staticmethod
    def _unpack(th):                           # s1 = sigma^2 (spatial variance), l1 = range, tau = tau^2 (nugget VARIANCE)
        return {"s1": np.exp(th[0]), "l1": np.exp(th[1]), "tau": np.exp(th[2])}

    def _nlp(self, th, ret=False):
        """Negative log posterior: restricted negative log-likelihood (GLS mean) + negative log prior (up to constants).
        With ret=True also returns the Cholesky factor F of K, the GLS mean beta and alpha = K^-1 (y - beta)."""
        p = self._unpack(th); n = len(self.y)
        K = p["s1"] * matern32(self.D, p["l1"]) + np.diag(np.full(n, p["tau"])) + S_.JITTER * np.eye(n)
        try:
            F = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        H = self.H; KiH = cho_solve(F, H); A = H.T @ KiH; beta = np.linalg.solve(A, KiH.T @ self.y)
        res = self.y - H @ beta; al = cho_solve(F, res)
        # restricted likelihood: 0.5 res' K^-1 res + 0.5 log|K| + 0.5 log|1' K^-1 1|
        nll = 0.5 * (res @ al) + np.log(np.diag(F[0])).sum() + 0.5 * np.linalg.slogdet(A)[1]
        # prior: Gaussian on log(range / extent) and on logit(spatial share)
        share = p["s1"] / (p["s1"] + p["tau"])
        dl = (np.log(p["l1"] / self.ext) - self.prior["mu_r"]) / self.prior["s_r"]
        ds = (np.log(share / (1 - share + 1e-12)) - self.prior["mu_s"]) / self.prior["s_s"]
        nll += 0.5 * (dl ** 2 + ds ** 2)
        return (nll, F, beta, al) if ret else nll

    def fit(self, coords, residuals, min_range):
        """Fit to residuals r at training coordinates; min_range = lower bound of the range (point_range_floor / block_range_floor)."""
        self.S = np.asarray(coords, float); r = np.asarray(residuals, float)
        self.ym, self.ys = float(np.mean(r)), float(np.std(r)) or 1.0           # standardise the residuals
        self.y = (r - self.ym) / self.ys
        self.D = cdist(self.S, self.S); self.H = np.ones((len(r), 1))
        mu_l = np.log(self.ext) + self.prior["mu_r"]                          # prior median of log(range)
        lv = (np.log(S_.VARIANCE_BOUNDS[0]), np.log(S_.VARIANCE_BOUNDS[1]))
        bounds = [lv, (np.log(min_range), mu_l + S_.RANGE_UPPER_LOG_OFFSET), lv]
        # 1) posterior mode from three starting ranges
        best = None
        for off in S_.MAP_START_LOG_OFFSETS:
            x0 = np.array([np.log(S_.MAP_START_SIGMA2), np.clip(mu_l + off, *bounds[1]), np.log(S_.MAP_START_TAU2)])
            res = minimize(self._nlp, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": S_.MAP_MAXITER})
            if best is None or res.fun < best.fun:
                best = res
        base = best.x; self.map_theta = base
        # 2) averaging grid around the mode: 7 ranges x 5 spatial shares, total variance fixed at the mode. Ranges outside the bounds are
        #    clipped to the bound, so near the lower bound several grid points coincide (kept, as in the paper's runs).
        grid, lp = [], []
        for dl in S_.GRID_LOG_RANGE_OFFSETS:
            for share in S_.GRID_SPATIAL_SHARES:
                th = base.copy(); tot = np.exp(base[0]) + np.exp(base[2])
                th[0] = np.log(share * tot); th[2] = np.log((1 - share) * tot); th[1] = np.clip(base[1] + dl, *bounds[1])
                grid.append(th); lp.append(-self._nlp(th))
        w = np.exp(np.array(lp) - max(lp))
        self.thetas, self.w = grid, w / w.sum()
        return self

    def _vbar_theta(self, F, p):
        Ki = cho_solve(F, np.eye(len(self.y)))
        return np.mean(np.clip(1 / np.diag(Ki) - p["tau"], 0, None))

    def predict_points(self, coords_q):
        """Latent mean mu_g, latent variance v_g (law of total variance over the grid) and v_bar, in the original units."""
        Sq = np.asarray(coords_q, float); Dq = cdist(Sq, self.S); Hq = np.ones((len(Sq), 1))
        mus, vs, vbars = [], [], []
        for th in self.thetas:
            p = self._unpack(th); _, F, beta, al = self._nlp(th, ret=True)
            kq = p["s1"] * matern32(Dq, p["l1"])
            Kikq = cho_solve(F, kq.T)
            var = p["s1"] - np.sum(kq * Kikq.T, 1)                              # simple-kriging variance ...
            R = Hq - (self.H.T @ Kikq).T; A = self.H.T @ cho_solve(F, self.H)
            var = var + np.sum(R @ np.linalg.inv(A) * R, 1)                     # ... plus the uncertainty of the GLS mean
            mus.append(Hq @ beta + kq @ al); vs.append(np.clip(var, 0, None)); vbars.append(self._vbar_theta(F, p))
        w = self.w[:, None]; mus = np.array(mus)
        m = (w * mus).sum(0); v = (w * (np.array(vs) + mus ** 2)).sum(0) - m ** 2
        return self.ym + self.ys * m, self.ys ** 2 * np.clip(v, 0, None), self.ys ** 2 * float(np.dot(self.w, vbars))

    def block_moments(self, coords_q, blocks):
        """Joint posterior of the latent field averaged over blocks.

        blocks: {name: indices into coords_q}. For each grid theta the posterior covariance of the latent field at the query locations is
            C = sigma^2 R** - sigma^4 R* K^-1 R*' + u u' / (1' K^-1 1),   u = 1 - sigma^2 R* K^-1 1   (last term: GLS-mean uncertainty),
        and a block's mean and variance are the averages of the posterior mean and of C over the block. Returns
        ({name: (block mean, block variance)} after mixing over the grid, v_bar), in the original units."""
        Sq = np.asarray(coords_q, float); Dq = cdist(Sq, self.S); Dqq = cdist(Sq, Sq); per_theta = []
        for th in self.thetas:
            p = self._unpack(th); _, F, beta, al = self._nlp(th, ret=True)
            kq = p["s1"] * matern32(Dq, p["l1"]); Kqq = p["s1"] * matern32(Dqq, p["l1"])
            Kikq = cho_solve(F, kq.T); C0 = Kqq - kq @ Kikq
            A = float((self.H.T @ cho_solve(F, self.H)).ravel()[0]); u = 1.0 - (self.H.T @ Kikq).ravel(); C = C0 + np.outer(u, u) / A
            mu = float(np.ravel(beta)[0]) + kq @ al
            per_theta.append((self._vbar_theta(F, p), {b: (float(mu[i].mean()), float(C[np.ix_(i, i)].mean())) for b, i in blocks.items()}))
        vbar = float(np.dot(self.w, [t[0] for t in per_theta])); out = {}
        for b in blocks:
            m_t = np.array([t[1][b][0] for t in per_theta]); v_t = np.array([t[1][b][1] for t in per_theta]); mb = float(np.dot(self.w, m_t))
            out[b] = (self.ym + self.ys * mb, self.ys ** 2 * (float(np.dot(self.w, v_t + m_t ** 2)) - mb ** 2))
        return out, self.ys ** 2 * vbar

    def hyperparameters(self):
        """MAP hyperparameters in the original units (range in metres)."""
        p = self._unpack(self.map_theta)
        return {"range_m": float(p["l1"]), "spatial_share": float(p["s1"] / (p["s1"] + p["tau"])),
                "spatial_variance": float(p["s1"] * self.ys ** 2), "nugget": float(p["tau"] * self.ys ** 2)}
