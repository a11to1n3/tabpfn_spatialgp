"""TabPFNSpatialGP: TabPFN with a spatial Gaussian process on its out-of-fold errors.

    y(x, s) = TabPFN(x, s | D) + g(s) + eps

Fitting (fit):
  1. Covariates are prepared (spectra -> principal components) and the projected coordinates are appended (preprocess.py).
  2. TabPFN is refitted on 5 folds of the training rows; each row's out-of-fold mean gives the residual r_i = y_i - m_T^(-k(i)).
  3. TabPFN is fitted on all training rows (the final model used for prediction).
  4. Point GP (n >= 25 only): residual GP with the property's empirical-Bayes prior (priors.POINT_PRIORS).
  5. Block GP (n >= 3): residual GP with the population prior (priors.BLOCK_PRIOR), used for field and zone means.

Prediction:
  predict        point predictions. For n >= 25, TabPFN's distribution is shifted by the GP mean mu_g and convolved with
                 N(0, max(0, v_g - v_bar)); for n < 25, TabPFN's own distribution is returned.
  predict_blocks field and zone means with uncertainty that accounts for the spatial correlation of errors (blocks.py).

Settings are in settings.py; the method, its evidence and its limits are in the README. Built with TabPFN.
"""
import functools
import warnings

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from . import settings as S_
from .backbone import make_regressor
from .bars import GaussianMixture, TabPFNBars, quantile_levels
from .blocks import as_index_blocks, block_predictive
from .gp_field import ResidualField, block_range_floor, field_extent, point_range_floor
from .preprocess import CoordinateTransform, CovariateTransform
from .priors import BLOCK_PRIOR, point_prior


def _load_backends():
    """Import torch and tabpfn before the thread limit is entered: threadpoolctl only limits libraries that are already loaded."""
    try:
        import torch  # noqa: F401
        import tabpfn  # noqa: F401
    except ImportError:
        pass


def _fixed_threads(method):
    """Run with settings.BLAS_THREADS (= 1) BLAS/OpenMP threads. TabPFN's output depends on the thread count for some inputs, and the
    paper's results were computed single-threaded."""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        _load_backends()
        with threadpool_limits(limits=S_.BLAS_THREADS):
            return method(*args, **kwargs)
    return wrapper


class TabPFNSpatialGP:
    """Parameters
    target_kind: 'SOC' (or organic matter), 'Clay' or 'pH'; selects the point model's prior. Needed only when n >= 25.
    point_prior: optional prior dict that overrides target_kind (not validated).
    device: 'auto' (the GPU if there is one, otherwise the CPU), 'cuda' or 'cpu'; model_path: TabPFN checkpoint ('auto' = its default).
    """

    def __init__(self, target_kind=None, point_prior=None, device="auto", model_path="auto"):
        self.target_kind, self.point_prior_override = target_kind, point_prior
        self.device, self.model_path = device, model_path

    def _regressor(self):
        return make_regressor(S_.RANDOM_STATE, self.device, self.model_path)

    # ------------------------------------------------------------------------------------------------------------------------- fitting
    @_fixed_threads
    def fit(self, X, coords, y, field_coords=None, spectral_mask=None, crs="projected"):
        """X: (n, p) covariates; coords: (n, 2) coordinates; y: (n,) measured property.
        field_coords: coordinates of all sample locations of the field (training and prediction). They set the field extent E, which
        scales the GP priors; the paper used all locations. If omitted, the training locations are used (with a warning).
        spectral_mask: boolean mask of spectral columns of X (see preprocess.spectral_mask_from_columns)."""
        X = np.asarray(X, float); coords = np.asarray(coords, float); y = np.asarray(y, float); n = len(y)
        if len(X) != n or len(coords) != n:
            raise ValueError(f"X ({len(X)} rows), coords ({len(coords)}) and y ({n}) must have the same length")
        if n < S_.BLOCK_GP_MIN_N:
            raise ValueError(f"at least {S_.BLOCK_GP_MIN_N} training samples are needed (got {n})")
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(coords))):
            raise ValueError("y and coords must be finite")
        if np.ptp(y) == 0:
            raise ValueError("y is constant: there is nothing to model")
        prior = None
        if n >= S_.POINT_GP_MIN_N:                                     # resolve the point prior before any TabPFN work
            prior = self.point_prior_override or point_prior(self.target_kind)
        if field_coords is None:
            warnings.warn("field_coords not given: the field extent is taken from the training locations only, "
                          "whereas the paper used all sample locations of the field")
        self.ct = CoordinateTransform(crs).fit(coords); self.S_train = self.ct.transform(coords)
        self.xt = CovariateTransform(spectral_mask).fit(X)
        self.X_train = np.column_stack([self.xt.transform(X), self.S_train])          # covariates, then coordinates
        self.y, self.n = y, n
        self.extent = field_extent(self.ct.transform(field_coords) if field_coords is not None else self.S_train)

        # step 2: out-of-fold TabPFN means -> residuals
        self.m_oof = np.empty(n)
        for a, b in KFold(min(S_.OOF_FOLDS, n), shuffle=True, random_state=S_.RANDOM_STATE).split(self.X_train):
            if np.ptp(y[a]) == 0:                                      # constant training fold (possible with rounded lab values): TabPFN 2.0.9
                self.m_oof[b] = y[a][0]; continue                      # collapses its bars to zero width, so predict the constant, as the
                                                                       # study's code does (experiments/core/point_model.py).
            self.m_oof[b] = TabPFNBars.from_tabpfn(self._regressor().fit(self.X_train[a], y[a]), self.X_train[b]).moments()[0]
        self.residuals = y - self.m_oof

        # step 3: final TabPFN on all training rows
        self.tabpfn = self._regressor().fit(self.X_train, y)

        # step 4: point GP (only from 25 points; below that it does not improve point predictions)
        self.point_gp = None
        if prior is not None:
            self.point_gp = ResidualField(prior, self.extent).fit(self.S_train, self.residuals, point_range_floor(self.S_train, prior, self.extent))

        # step 5: block GP (population prior, from 3 points; evaluated from 5)
        self.block_gp = ResidualField(BLOCK_PRIOR, self.extent).fit(self.S_train, self.residuals, block_range_floor(self.S_train))
        return self

    # ---------------------------------------------------------------------------------------------------------------------- prediction
    def _tabpfn_at(self, Xq, coords_q):
        Sq = self.ct.transform(coords_q)
        return TabPFNBars.from_tabpfn(self.tabpfn, np.column_stack([self.xt.transform(Xq), Sq])), Sq

    @_fixed_threads
    def predict(self, Xq, coords_q, quantiles=(0.1, 0.5, 0.9)):
        """Point predictions at new locations. Returns dict(mean, quantiles {level: values}, gp_correction, distribution); the
        distribution object has moments(), quantiles(levels) and crps(y). 'mean' is TabPFN's exact mean plus mu_g, as in the paper's
        runs; for n >= 25 it can differ slightly from distribution.moments()[0], which uses the 400-quantile representation."""
        bars, Sq = self._tabpfn_at(Xq, coords_q)
        mean_T = bars.moments()[0]
        if self.point_gp is None:                                       # n < 25: TabPFN alone
            dist, mu = bars, np.zeros(len(Sq))
        else:                                                           # shift by mu_g and convolve with N(0, max(0, v_g - v_bar))
            mu, v_g, v_bar = self.point_gp.predict_points(Sq)
            dist = GaussianMixture(bars.quantiles(quantile_levels(S_.N_MIXTURE)) + mu[:, None], np.sqrt(np.clip(v_g - v_bar, 0, None)))
        q = dist.quantiles(np.asarray(quantiles, float))
        return {"mean": mean_T + mu, "quantiles": {lv: q[:, j] for j, lv in enumerate(quantiles)}, "gp_correction": mu, "distribution": dist}

    @_fixed_threads
    def predict_blocks(self, Xq, coords_q, blocks=None, kappa=None):
        """Means over blocks of prediction locations (a field, management or k-means zones) with their uncertainty.
        blocks: {name: indices or boolean mask over the prediction locations}; default: one block with all of them ('field'). Use
        blocks.zone_blocks(...) for the study's zones. kappa: factor on the block standard deviation; default 1.15
        (settings.KAPPA, the study's value). Returns {name: dict(mean, sd, interval80, ...)};
        see blocks.block_predictive.
        Memory grows with the square of the number of prediction locations (the joint posterior): use sample points, not dense rasters."""
        bars, Sq = self._tabpfn_at(Xq, coords_q)
        blocks = as_index_blocks({"field": np.arange(len(Sq))} if blocks is None else blocks, len(Sq))
        gp_moments, v_bar = self.block_gp.block_moments(Sq, blocks)
        return block_predictive(bars.moments()[0], bars.interval80_variance(), gp_moments, v_bar, self.n, blocks,
                                S_.KAPPA if kappa is None else float(kappa))

    # --------------------------------------------------------------------------------------------------------------------- diagnostics
    def report(self):
        """What was fitted, with which settings and hyperparameters."""
        rep = {"n_train": int(self.n), "field_extent_m": self.extent, "coordinate_mode": self.ct.mode,
               "point_gp": None if self.point_gp is None else {"prior": self.point_gp.prior, **self.point_gp.hyperparameters()},
               "block_gp": {"prior": self.block_gp.prior, **self.block_gp.hyperparameters()},
               "oof_rmse_tabpfn": float(np.sqrt(np.mean(self.residuals ** 2))),
               "settings": {"n_estimators": S_.N_ESTIMATORS, "tabpfn": S_.TABPFN_VERSION, "point_gp_min_n": S_.POINT_GP_MIN_N,
                            "block_mean_correction_min_n": S_.BLOCK_MEAN_CORRECTION_MIN_N, "kappa": S_.KAPPA}}
        if self.point_gp is None:
            rep["point_note"] = f"n = {self.n} < {S_.POINT_GP_MIN_N}: point predictions are TabPFN's own"
        return rep
