"""Input preparation.

Covariates: spectral columns are standardised and reduced to principal components fitted on the training rows (10 components, n - 1
when n <= 10). All other covariates pass through unchanged, because TabPFN handles scaling itself. This matches covariates() in experiments/core/common.py,
whose results are identical when there are no missing values.

Coordinates: projected metres are expected (crs='projected', the default), as in the study. With crs='lonlat', or crs='auto' and
coordinates that look like degrees, WGS84 longitude/latitude is projected with a local equirectangular projection on a sphere (scale
error below 1 %, which hardly matters because the GP priors scale with the field extent). The paper's results used projected
coordinates only.
"""
import warnings
import numpy as np
from sklearn.decomposition import PCA

from . import settings as S_

EARTH_RADIUS = 6371008.8


def spectral_mask_from_columns(columns, prefixes=("wl_",)):
    """Boolean mask of spectral columns. LimeSoDa names spectral columns 'wl_<wavelength>' (the only prefix used in the study)."""
    return np.array([any(str(c).startswith(p) for p in prefixes) for c in columns])


def looks_like_lonlat(coords):
    c = np.asarray(coords, float)
    return bool(np.all(np.abs(c[:, 0]) <= 180) and np.all(np.abs(c[:, 1]) <= 90) and np.ptp(c[:, 0]) < 1 and np.ptp(c[:, 1]) < 1)


class CoordinateTransform:
    """Fitted on the training coordinates and applied identically to prediction coordinates.
    crs: 'projected' (metres, default), 'lonlat', 'auto' or 'EPSG:xxxx' (needs pyproj)."""

    def __init__(self, crs="projected"):
        self.crs = crs

    def fit(self, coords):
        c = np.asarray(coords, float)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError("coords must be an (n, 2) array")
        mode = self.crs
        if not (mode in ("projected", "lonlat", "auto") or (isinstance(mode, str) and mode.upper().startswith("EPSG:"))):
            raise ValueError(f"crs must be 'projected', 'lonlat', 'auto' or 'EPSG:xxxx', not {mode!r}")
        if mode == "auto":
            mode = "lonlat" if looks_like_lonlat(c) else "projected"
        if mode == "projected" and looks_like_lonlat(c):
            raise ValueError("coordinates look like longitude/latitude degrees; pass crs='lonlat' or project them to metres first")
        if isinstance(mode, str) and mode.upper().startswith("EPSG:"):
            import pyproj
            crs = pyproj.CRS(mode)
            if crs.is_projected:
                if crs.axis_info[0].unit_name not in ("metre", "meter"):
                    raise ValueError(f"projected CRS unit is {crs.axis_info[0].unit_name}; metres are required")
                mode = "projected"
            else:
                mode = "lonlat"
        self.mode = mode
        if mode == "lonlat":
            self.lon0, self.lat0 = float(c[:, 0].mean()), float(c[:, 1].mean())
        return self

    def transform(self, coords):
        c = np.asarray(coords, float)
        if self.mode == "lonlat":
            x = EARTH_RADIUS * np.cos(np.radians(self.lat0)) * np.radians(c[:, 0] - self.lon0)
            y = EARTH_RADIUS * np.radians(c[:, 1] - self.lat0)
            return np.column_stack([x, y])
        return c


class CovariateTransform:
    """Spectral columns -> standardised -> min(10, n_train - 1) principal components fitted on the training rows; other columns unchanged.
    Output column order: non-spectral covariates, then the principal components (as in the study)."""

    def __init__(self, spectral_mask=None, n_components=S_.N_SPECTRAL_PCS):
        self.spectral_mask, self.n_components = spectral_mask, n_components

    def fit(self, X):
        X = np.asarray(X, float); m = self.spectral_mask
        self.m = np.zeros(X.shape[1], bool) if m is None else np.asarray(m, bool)
        if self.m.any():
            Xs = X[:, self.m]
            if np.isnan(Xs).any():
                warnings.warn("missing values in spectral columns are mean-imputed before PCA (the study data had none)")
            self.mu = np.nanmean(Xs, 0); self.sd = np.nanstd(Xs, 0, ddof=1); self.sd[~(self.sd > 0)] = 1
            k = int(min(self.n_components, len(X) - 1, Xs.shape[1]))
            self.pca = PCA(k, svd_solver="full").fit(np.nan_to_num((Xs - self.mu) / self.sd))
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        if not self.m.any():
            return X
        Z = self.pca.transform(np.nan_to_num((X[:, self.m] - self.mu) / self.sd))
        return np.column_stack([X[:, ~self.m], Z])
