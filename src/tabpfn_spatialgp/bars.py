"""Predictive distributions.

TabPFNBars: TabPFN v2's own predictive distribution for regression ("bar distribution"). It is piecewise uniform on interior bars with
half-normal tails on the two end bars. Moments, quantiles and CRPS are computed exactly from TabPFN's bar borders and logits.

GaussianMixture: the point predictive when the GP is used (n >= 25). TabPFN's distribution is shifted by the GP mean correction mu_g and
convolved with N(0, max(0, v_g - v_bar)). It is represented as an equal-weight mixture of Gaussians centred on 400 equally spaced
quantiles of TabPFN's distribution, all with the same standard deviation.

Both classes expose the same methods: moments() -> (mean, variance), quantiles(probabilities) -> (n, k), crps(y) -> (n,).
"""
import numpy as np
from scipy.special import logsumexp, erfc
from scipy.stats import norm


class TabPFNBars:
    """TabPFN's bar distribution for n query points: borders (B+1,) shared by all points, logits (n, B)."""

    def __init__(self, borders, logits):
        self.borders = np.asarray(borders, dtype=float)
        self.logits = np.asarray(logits, dtype=float)
        self.weights = np.exp(self.logits - logsumexp(self.logits, axis=-1, keepdims=True))     # bar probabilities
        self.widths = np.diff(self.borders)
        if not np.all(self.widths > 0) or self.weights.shape[-1] != len(self.widths):
            raise ValueError("Invalid bar specification")
        self.scales = self.widths[[0, -1]] / norm.ppf(.75)        # half-normal scales of the two tails (median = bar width)
        self.cumulative = np.concatenate([np.zeros((len(self.weights), 1)), np.cumsum(self.weights, axis=1)], axis=1)

    @classmethod
    def from_tabpfn(cls, regressor, X):
        """Predictive distribution of a fitted tabpfn.TabPFNRegressor at rows X."""
        out = regressor.predict(X, output_type="full", quantiles=[0.5])
        return cls(out["criterion"].borders.detach().cpu().numpy(), out["logits"].detach().cpu().numpy())

    def quantiles(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        if np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValueError("Quantiles must be strictly between zero and one")
        result = np.empty((len(self.weights), len(probabilities)))
        for row, weights in enumerate(self.weights):
            indices = np.clip(np.searchsorted(self.cumulative[row, 1:], probabilities), 0, len(weights) - 1)
            q = self.borders[indices] + self.widths[indices] * (probabilities - self.cumulative[row, indices]) / weights[indices]
            left, right = indices == 0, indices == len(weights) - 1          # quantiles in the tails: invert the half-normals
            q[left] = self.borders[1] - self.scales[0] * norm.ppf(1 - probabilities[left] / (2 * weights[0]))
            q[right] = self.borders[-2] + self.scales[1] * norm.ppf(1 - (1 - probabilities[right]) / (2 * weights[-1]))
            result[row] = q
        return result

    def moments(self):
        """Exact mean and variance (uniform interior bars, half-normal tails)."""
        means = (self.borders[:-1] + self.borders[1:]) / 2
        squares = (self.borders[:-1] ** 2 + self.borders[:-1] * self.borders[1:] + self.borders[1:] ** 2) / 3
        halfnormal_mean = self.scales * np.sqrt(2 / np.pi)
        means[0] = self.borders[1] - halfnormal_mean[0]
        means[-1] = self.borders[-2] + halfnormal_mean[1]
        squares[0] = self.scales[0] ** 2 * (1 - 2 / np.pi) + means[0] ** 2
        squares[-1] = self.scales[1] ** 2 * (1 - 2 / np.pi) + means[-1] ** 2
        mean = self.weights @ means
        return mean, self.weights @ squares - mean ** 2

    def crps(self, y):
        """Exact CRPS of the observations y (one per query point)."""
        y = np.asarray(y, dtype=float)
        # exact integral on every piecewise-linear interior CDF segment
        lower, upper = self.borders[1:-2], self.borders[2:-1]
        widths = upper - lower
        fraction = np.clip((y[:, None] - lower) / widths, 0, 1)
        f0 = self.cumulative[:, 1:-2]
        f1 = f0 + self.weights[:, 1:-1] * fraction
        f2 = self.cumulative[:, 2:-1]
        result = np.sum(widths * fraction * (f0 ** 2 + f0 * f1 + f1 ** 2) / 3, axis=1)
        result += np.sum(widths * (1 - fraction) * ((1 - f1) ** 2 + (1 - f1) * (1 - f2) + (1 - f2) ** 2) / 3, axis=1)
        # tails: integral of erfc(t / sqrt(2))^2 from zero to infinity
        constant = 2 * (np.sqrt(2) - 1) / np.sqrt(np.pi)
        for weight, scale, h in [(self.weights[:, 0], self.scales[0], np.maximum((self.borders[1] - y) / self.scales[0], 0)),
                                 (self.weights[:, -1], self.scales[1], np.maximum((y - self.borders[-2]) / self.scales[1], 0))]:
            integral = h * erfc(h / np.sqrt(2)) + np.sqrt(2 / np.pi) * (-np.expm1(-h ** 2 / 2))
            result += scale * (weight ** 2 * constant + h - 2 * weight * integral)
        return result

    def interval80_variance(self):
        """v_T: Gaussian-equivalent variance of the central 80 % interval, ((Q(0.9) - Q(0.1)) / (2 z_0.9))^2."""
        from .settings import Z80
        q = self.quantiles([0.1, 0.9])
        return ((q[:, 1] - q[:, 0]) / (2 * Z80)) ** 2


def quantile_levels(n):
    """n equally spaced probability levels (i + 0.5) / n."""
    return (np.arange(n) + 0.5) / n


class GaussianMixture:
    """Equal-weight Gaussian mixture per query point: centres (n, K), one standard deviation per point (n,)."""

    def __init__(self, centers, sd):
        self.centers = np.asarray(centers, float)
        self.sd = np.maximum(np.asarray(sd, float), 1e-12)

    def moments(self):
        return self.centers.mean(1), self.centers.var(1) + self.sd ** 2

    def cdf(self, y):
        y = np.asarray(y, float)
        return norm.cdf((y[:, None] - self.centers) / self.sd[:, None]).mean(1)

    def quantiles(self, probabilities, iters=60):
        """Quantiles by bisection on the mixture CDF (60 halvings of a bracket of +-8 sd around the centres)."""
        probabilities = np.atleast_1d(np.asarray(probabilities, float)); s = self.sd
        lo, hi = self.centers.min(1) - 8 * s, self.centers.max(1) + 8 * s
        out = np.empty((len(self.centers), len(probabilities)))
        for j, p in enumerate(probabilities):
            a, b = lo.copy(), hi.copy()
            for _ in range(iters):
                m = (a + b) / 2; c = norm.cdf((m[:, None] - self.centers) / s[:, None]).mean(1); a = np.where(c < p, m, a); b = np.where(c < p, b, m)
            out[:, j] = (a + b) / 2
        return out

    def crps(self, y):
        """Exact CRPS of a Gaussian mixture: E|X - y| - 0.5 E|X - X'|."""
        y = np.asarray(y, float); s = self.sd[:, None]

        def A(mu, sig):                                  # E|Z| for Z ~ N(mu, sig^2)
            z = mu / sig
            return mu * (2 * norm.cdf(z) - 1) + 2 * sig * norm.pdf(z)

        t1 = A(self.centers - y[:, None], s).mean(1)
        t2 = np.array([A(c[:, None] - c[None, :], np.sqrt(2) * si).mean() for c, si in zip(self.centers, s[:, 0])])
        return t1 - 0.5 * t2
