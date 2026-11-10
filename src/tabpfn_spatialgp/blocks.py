"""Field and zone means: blocks and their predictive distribution (the paper's block model, kappa = 1.15).

For a block B of m prediction locations the target is the mean of the property over B. With m_T and v_T from TabPFN (v_T = Gaussian-
equivalent variance of TabPFN's central 80 % interval), and the block GP's joint posterior (gp_field.ResidualField.block_moments):

    mean     = (1/m) sum_B m_T  +  [n >= 25] * mu_gB                                  (GP mean correction only from 25 training points)
    variance = kappa^2 * ( V_gB  +  (1/m^2) sum_B max(0, v_T - v_bar) )               (correlated part from the GP + independent part)

and the predictive is Gaussian. Treating TabPFN's errors as independent would give only (1/m^2) sum_B v_T, which shrinks like 1/m and is
far too narrow when errors are spatially correlated.
"""
import numpy as np
from scipy.stats import norm

from . import settings as S_


def zone_blocks(field_coords, query_index, k=S_.ZONE_K, min_points=S_.ZONE_MIN_POINTS):
    """The blocks used in the study.

    field_coords: coordinates of ALL sample locations of the field (training and prediction), projected (metres); at least 4.
    query_index:  positions of the prediction locations within field_coords.
    Returns {'field': all prediction locations, 'zone<j>': prediction locations in k-means cluster j (if at least min_points)}, as
    indices into the prediction locations."""
    from sklearn.cluster import KMeans
    labels = KMeans(k, random_state=0, n_init=10).fit_predict(np.asarray(field_coords, float))
    q = np.asarray(query_index)
    blocks = {"field": np.arange(len(q))}
    for j in range(k):
        idx = np.flatnonzero(labels[q] == j)
        if len(idx) >= min_points:
            blocks[f"zone{j}"] = idx
    return blocks


def as_index_blocks(blocks, n_points):
    """Normalise {name: indices or boolean mask} to {name: integer indices}; reject empty or out-of-range blocks."""
    out = {}
    for b, i in blocks.items():
        i = np.asarray(i)
        if i.dtype == bool:
            if len(i) != n_points:
                raise ValueError(f"block {b!r}: boolean mask has length {len(i)}, expected {n_points}")
            i = np.flatnonzero(i)
        i = i.astype(int)
        if len(i) == 0:
            raise ValueError(f"block {b!r} is empty")
        if i.min() < 0 or i.max() >= n_points:
            raise ValueError(f"block {b!r} has indices outside 0..{n_points - 1}")
        out[b] = i
    return out


def block_predictive(m_T, v_T, gp_moments, v_bar, n_train, blocks, kappa=S_.KAPPA):
    """Combine TabPFN's pointwise outputs with the block GP into Gaussian block predictives.

    m_T, v_T:   TabPFN mean and 80 %-interval variance at the prediction locations.
    gp_moments: {block: (mu_gB, V_gB)} from ResidualField.block_moments; v_bar from the same call.
    Returns {block: dict(mean, sd, interval80, var_correlated, var_independent, sd_if_independent)}."""
    m_T = np.asarray(m_T, float); v_T = np.asarray(v_T, float); blocks = as_index_blocks(blocks, len(m_T))
    excess = np.clip(v_T - v_bar, 0, None)                  # TabPFN variance in excess of what the GP already represents
    out = {}
    for b, i in blocks.items():
        m = len(i); mu_gB, V_gB = gp_moments[b]
        mean = float(m_T[i].mean()) + (mu_gB if n_train >= S_.BLOCK_MEAN_CORRECTION_MIN_N else 0.0)
        v_ind = float(np.sum(excess[i])) / m ** 2
        sd = kappa * float(np.sqrt(max(V_gB + v_ind, 1e-18)))
        out[b] = {"mean": mean, "sd": sd, "interval80": (mean - S_.Z80 * sd, mean + S_.Z80 * sd), "var_correlated": V_gB,
                  "var_independent": v_ind, "sd_if_independent": float(np.sqrt(np.sum(v_T[i]))) / m}
    return out


def gaussian_crps(mean, sd, y):
    """CRPS of N(mean, sd^2) at y (used to score block predictions)."""
    z = (y - mean) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
