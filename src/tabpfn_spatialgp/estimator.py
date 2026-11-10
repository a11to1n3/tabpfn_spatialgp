"""TabPFNGPRegressor: the scikit-learn interface to TabPFN-GP.

The coordinates are two columns of X (coord_cols), so the estimator follows the scikit-learn contract:
  - fit(X, y) returns self, and predict(X) returns an array of means;
  - get_params, set_params and clone work;
  - it can be cross-validated, put in a Pipeline and tuned with GridSearchCV.

Field and zone means, the method's purpose, come from predict_blocks. Every computation is delegated to TabPFNSpatialGP with the same
settings, so the outputs equal those of the core class (tests/test_estimator.py).
"""
import inspect

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from . import settings as S_
from .model import TabPFNSpatialGP

# missing covariates are allowed (TabPFN handles them); missing coordinates are rejected by TabPFNSpatialGP.fit
_FINITE = ("ensure_all_finite" if "ensure_all_finite" in inspect.signature(check_array).parameters else "force_all_finite")


class TabPFNGPRegressor(RegressorMixin, BaseEstimator):
    """TabPFN-GP as a scikit-learn regressor.

    Parameters
    coord_cols: the two columns of X with the coordinates, by name (X a DataFrame) or position; default: the last two.
    spectral_cols: columns of X with spectra, reduced to principal components as in the study (names or positions); default: none.
    target_kind: 'SOC' (or organic matter), 'Clay' or 'pH': selects the point model's prior, needed only from 25 training samples.
    point_prior: optional prior dict that overrides target_kind.
    kappa: factor on the standard deviation of field and zone means; None: settings.KAPPA (1.15, the study's value).
    crs: 'projected' (metres), 'lonlat', 'auto' or 'EPSG:xxxx' (needs pyproj).
    device: 'auto' (the GPU if there is one, otherwise the CPU), 'cuda' or 'cpu'; model_path: TabPFN checkpoint ('auto' = its default).

    Fitted attributes
    model_: the fitted TabPFNSpatialGP; n_features_in_, feature_names_in_ (X a DataFrame with string column names);
    coord_idx_, covariate_idx_: positions of the coordinate and covariate columns; spectral_mask_: spectral columns among the covariates.
    """

    def __init__(self, coord_cols=(-2, -1), spectral_cols=None, target_kind=None, point_prior=None, kappa=None, crs="projected",
                 device="auto", model_path="auto"):
        self.coord_cols = coord_cols
        self.spectral_cols = spectral_cols
        self.target_kind = target_kind
        self.point_prior = point_prior
        self.kappa = kappa
        self.crs = crs
        self.device = device
        self.model_path = model_path

    # --------------------------------------------------------------------------------------------------------------------- columns
    def _split(self, X, fitting):
        """Covariates and coordinates of X, by the columns set at fit."""
        names = [str(c) for c in X.columns] if hasattr(X, "columns") else None
        arr = check_array(X, dtype=np.float64, ensure_min_features=2 if fitting else 1,
                          ensure_min_samples=S_.BLOCK_GP_MIN_N if fitting else 1,
                          **{_FINITE: "allow-nan"})
        if fitting:
            p = arr.shape[1]; self.n_features_in_ = p
            if names is not None and all(isinstance(c, str) for c in X.columns):
                self.feature_names_in_ = np.asarray(names, dtype=object)
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_

            def positions(cols, what):
                out = []
                for c in ([] if cols is None else list(cols)):
                    if isinstance(c, str):
                        if names is None:
                            raise ValueError(f"{what} gives column {c!r} by name: pass X as a DataFrame, or give positions")
                        if c not in names:
                            raise ValueError(f"{what}: column {c!r} is not in X")
                        out.append(names.index(c))
                    else:
                        k = int(c)
                        if not -p <= k < p:
                            raise ValueError(f"{what}: position {k} is outside X's {p} columns")
                        out.append(k % p)
                return out
            coord = positions(self.coord_cols, "coord_cols")
            if len(coord) != 2 or coord[0] == coord[1]:
                raise ValueError("coord_cols must give two different columns (the projected coordinates)")
            spectral = positions(self.spectral_cols, "spectral_cols")
            if set(spectral) & set(coord):
                raise ValueError("a column cannot be both a coordinate and a spectrum")
            self.coord_idx_ = np.asarray(coord)
            self.covariate_idx_ = np.asarray([j for j in range(p) if j not in coord], dtype=int)
            self.spectral_mask_ = np.isin(self.covariate_idx_, spectral)
        else:
            check_is_fitted(self, "model_")
            if arr.shape[1] != self.n_features_in_:
                raise ValueError(f"X has {arr.shape[1]} features, but {type(self).__name__} is expecting {self.n_features_in_} features "
                                 "as input.")
            if hasattr(self, "feature_names_in_") and names is not None and names != list(self.feature_names_in_):
                raise ValueError("X's column names differ from those seen at fit")
        return arr[:, self.covariate_idx_], arr[:, self.coord_idx_]

    # ------------------------------------------------------------------------------------------------------------------------- fit
    def fit(self, X, y, field_coords=None):
        """X: covariates and the two coordinate columns; y: the measured property. field_coords: coordinates of all sampled locations
        of the field, which set the field extent that scales the GP's prior, as in the study (default: the training locations, with a
        warning)."""
        if y is None:
            raise ValueError(f"{type(self).__name__} requires y to be passed, but the target y is None.")
        Xc, coords = self._split(X, fitting=True)
        y = column_or_1d(check_array(y, ensure_2d=False, dtype=np.float64, **{_FINITE: True}), warn=True)
        if len(Xc) != len(y):
            raise ValueError(f"X ({len(Xc)} rows) and y ({len(y)}) must have the same length")
        self.model_ = TabPFNSpatialGP(target_kind=self.target_kind, point_prior=self.point_prior, device=self.device,
                                      model_path=self.model_path)
        self.model_.fit(Xc, coords, y, field_coords=field_coords, spectral_mask=self.spectral_mask_, crs=self.crs)
        return self

    # --------------------------------------------------------------------------------------------------------------------- predict
    def predict(self, X):
        """Point predictions: the mean at each row of X (TabPFN's, corrected by the GP from 25 training samples)."""
        Xc, coords = self._split(X, fitting=False)
        return self.model_.predict(Xc, coords)["mean"]

    def predict_quantiles(self, X, quantiles=(0.1, 0.5, 0.9)):
        """Quantiles of the point predictive distribution: an array of shape (n_rows, len(quantiles))."""
        Xc, coords = self._split(X, fitting=False)
        q = self.model_.predict(Xc, coords, quantiles=tuple(quantiles))["quantiles"]
        return np.column_stack([q[lv] for lv in quantiles])

    def predict_distribution(self, X):
        """The point predictive distributions, an object with moments(), quantiles(levels) and crps(y)."""
        Xc, coords = self._split(X, fitting=False)
        return self.model_.predict(Xc, coords)["distribution"]

    def predict_blocks(self, X, blocks=None):
        """Means over blocks of the rows of X (the field, zones) with their uncertainty, as in the paper.
        blocks: {name: row indices or boolean mask}; default: one block 'field' with all rows. Returns {name: dict(mean, sd,
        interval80, ...)} with kappa = self.kappa (default 1.15)."""
        Xc, coords = self._split(X, fitting=False)
        return self.model_.predict_blocks(Xc, coords, blocks=blocks, kappa=self.kappa)

    def report(self):
        """What was fitted, with which settings and hyperparameters."""
        check_is_fitted(self, "model_")
        return self.model_.report()

    # ------------------------------------------------------------------------------------------------------------------------- tags
    def __sklearn_tags__(self):                                          # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True                                 # TabPFN accepts missing covariates (not coordinates)
        return tags

    def _more_tags(self):                                                # scikit-learn < 1.6
        return {"allow_nan": True}
