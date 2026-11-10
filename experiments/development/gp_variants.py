"""The GP variants of the development screen. Used by development.py to refit them on the 14 development fields
(docs/development/plan_development.md, choice 4)."""
"""GP spatial-field variants for TabPFN residuals (development screen).
All variants: latent field g(s) on projected coordinates + noise, fitted to out-of-fold backbone residuals r_i, with the empirical-Bayes
transfer prior (log range/extent, logit spatial share). Options:
  het   : noise_i = tau2 + kappa * v_i/mean(v)  (v_i = backbone out-of-fold predictive variance: kriging with heteroscedastic, TabPFN-informed error)
  drift : universal kriging mean beta0 + beta1 x + beta2 y (GLS, coefficient uncertainty propagated)
  two   : Matern-3/2 short-range + Matern-3/2 long-range (range2 = 4..20 x range1)
  bma   : average over a grid of (range, spatial share) weighted by marginal likelihood x prior, instead of MAP
Returns latent mean/variance at query points and the mean LOO latent variance at training rows."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def m32(D, ell):
    r = np.sqrt(3.0) * D / ell; return (1 + r) * np.exp(-r)


class SpatialField:
    def __init__(self, het=False, drift=False, two=False, prior=None, ext=1.0, null=False, shrink=False):
        self.het, self.drift, self.two, self.prior, self.ext, self.null, self.shrink = het, drift, two, prior, ext, null, shrink

    # theta (log): [s1, l1, tau, (kappa), (s2, l2_mult)]
    def _unpack(self, th):
        p = {"s1": np.exp(th[0]), "l1": np.exp(th[1]), "tau": np.exp(th[2])}; i = 3
        if self.het: p["kappa"] = np.exp(th[i]); i += 1
        if self.two: p["s2"] = np.exp(th[i]); p["l2"] = p["l1"] * (4 + 16 / (1 + np.exp(-th[i + 1]))); i += 2
        return p

    def _K(self, p, D, n):
        K = p["s1"] * m32(D, p["l1"])
        if self.two: K = K + p["s2"] * m32(D, p["l2"])
        return K

    def _noise(self, p):
        nz = np.full(len(self.y), p["tau"])
        if self.het: nz = nz + p["kappa"] * self.v
        return nz

    def _H(self, S):
        h = [np.ones(len(S))]
        if self.drift: h += [(S[:, 0] - self.cx) / self.ext, (S[:, 1] - self.cy) / self.ext]
        return np.column_stack(h)

    def _nlp(self, th, ret=False):
        p = self._unpack(th); n = len(self.y)
        K = self._K(p, self.D, n) + np.diag(self._noise(p)) + 1e-8 * np.eye(n)
        try: F = cho_factor(K, lower=True)
        except np.linalg.LinAlgError: return 1e10
        H = self.H; KiH = cho_solve(F, H); A = H.T @ KiH; beta = np.linalg.solve(A, KiH.T @ self.y)
        res = self.y - H @ beta; al = cho_solve(F, res)
        nll = 0.5 * (res @ al) + np.log(np.diag(F[0])).sum() + 0.5 * np.linalg.slogdet(A)[1]          # REML-style restricted likelihood
        if self.prior is not None:
            share = (p["s1"] + p.get("s2", 0)) / (p["s1"] + p.get("s2", 0) + p["tau"])
            dl = (np.log(p["l1"] / self.ext) - self.prior["mu_r"]) / self.prior["s_r"]
            ds = (np.log(share / (1 - share + 1e-12)) - self.prior["mu_s"]) / self.prior["s_s"]
            nll += 0.5 * (dl ** 2 + ds ** 2)
        if ret: return nll, F, beta, al, K
        return nll

    def fit(self, S, r, v=None, min_len=None, bma=False):
        self.S = np.asarray(S, float); self.ym, self.ys = float(np.mean(r)), float(np.std(r)) or 1.0
        self.y = (np.asarray(r, float) - self.ym) / self.ys; self.v = (np.ones(len(r)) if v is None else np.asarray(v, float) / np.mean(v))
        self.cx, self.cy = self.S[:, 0].mean(), self.S[:, 1].mean(); self.D = cdist(self.S, self.S); self.H = self._H(self.S)
        mu_l = np.log(self.ext) + (self.prior["mu_r"] if self.prior else np.log(0.15)); lo = np.log(min_len) if min_len else mu_l - 5
        k = 3 + self.het + 2 * self.two
        bounds = [(np.log(1e-4), np.log(10)), (lo, mu_l + 4), (np.log(1e-4), np.log(10))] + ([(np.log(1e-4), np.log(10))] if self.het else []) + ([(np.log(1e-4), np.log(10)), (-4, 4)] if self.two else [])
        if not bma:
            best = None
            for l0 in (mu_l - 0.7, mu_l, mu_l + 0.7):
                x0 = np.array([np.log(0.3), np.clip(l0, *bounds[1]), np.log(0.7)] + ([np.log(0.3)] if self.het else []) + ([np.log(0.1), 0.0] if self.two else []))
                rr = minimize(self._nlp, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
                if best is None or rr.fun < best.fun: best = rr
            self.thetas, self.w = [best.x], np.array([1.0])
        else:                                                                   # grid over (range, share) at fixed total variance; other params at MAP
            base = self.fit(S, r, v, min_len, bma=False).thetas[0]; grid, lp = [], []
            for dl in np.linspace(-1.5, 1.5, 7):
                for share in (0.1, 0.25, 0.5, 0.75, 0.9):
                    th = base.copy(); tot = np.exp(base[0]) + np.exp(base[2]); th[0] = np.log(share * tot); th[2] = np.log((1 - share) * tot)
                    th[1] = np.clip(base[1] + dl, *bounds[1]); grid.append(th); lp.append(-self._nlp(th))
            lp = np.array(lp) + (np.log(0.5 / len(lp)) if self.null else 0.0)
            if self.null:                                                     # spike: no spatial field (share 1e-4), prior mass 0.5
                th = base.copy(); tot = np.exp(base[0]) + np.exp(base[2]); th[0] = np.log(1e-4 * tot); th[2] = np.log(tot); grid.append(th); lp = np.append(lp, -self._nlp(th) + np.log(0.5))
            w = np.exp(lp - lp.max()); self.thetas, self.w = grid, w / w.sum()
        return self

    def predict(self, Sq):
        """Latent mean/variance at Sq. With shrink=True the spatially varying part of the field (not the GLS mean) is multiplied by
        c = argmin_c sum_i (res_i - c * g_loo_i)^2 clipped to [0, 1], where g_loo are the closed-form leave-one-out field predictions at the
        training rows (empirical-Bayes calibration of how much of the kriged correction to trust)."""
        Sq = np.asarray(Sq, float); Dq = cdist(Sq, self.S); Hq = self._H(Sq); means, fields, vs, vbars, gl, rs = [], [], [], [], [], []
        for th in self.thetas:
            p = self._unpack(th); nll, F, beta, al, K = self._nlp(th, ret=True)
            kq = p["s1"] * m32(Dq, p["l1"]) + (p["s2"] * m32(Dq, p["l2"]) if self.two else 0)
            kqq = p["s1"] + p.get("s2", 0)
            Kikq = cho_solve(F, kq.T); var = kqq - np.sum(kq * Kikq.T, 1)
            R = Hq - (self.H.T @ Kikq).T; A = self.H.T @ cho_solve(F, self.H); var = var + np.sum(R @ np.linalg.inv(A) * R, 1)
            Ki = cho_solve(F, np.eye(len(self.y))); vb = np.mean(np.clip(1 / np.diag(Ki) - self._noise(p), 0, None))
            res = self.y - self.H @ beta
            means.append(Hq @ beta); fields.append(kq @ al); vs.append(np.clip(var, 0, None)); vbars.append(vb); gl.append(res - al / np.diag(Ki)); rs.append(res)
        w = self.w[:, None]; fields = np.array(fields); means = np.array(means); self.c = 1.0
        if self.shrink:
            g_loo = (w * np.array(gl)).sum(0); res = (w * np.array(rs)).sum(0)
            self.c = float(np.clip(np.dot(res, g_loo) / max(np.dot(g_loo, g_loo), 1e-12), 0, 1))
        mus = means + self.c * fields; vs = np.array(vs)
        m = (w * mus).sum(0); v = (w * (vs + mus ** 2)).sum(0) - m ** 2
        return self.ym + self.ys * m, self.ys ** 2 * np.clip(v, 0, None), self.ys ** 2 * float(np.dot(self.w, vbars))
