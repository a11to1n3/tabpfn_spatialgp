"""Core tests on synthetic data: no TabPFN, no GPU, run anywhere (python tests/run_tests.py, or pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabpfn_spatialgp import settings as S_                                   # noqa: E402
from tabpfn_spatialgp.bars import GaussianMixture, TabPFNBars                 # noqa: E402
from tabpfn_spatialgp.blocks import as_index_blocks, block_predictive, zone_blocks  # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent  # noqa: E402
from tabpfn_spatialgp.preprocess import CoordinateTransform                   # noqa: E402
from tabpfn_spatialgp.priors import BLOCK_PRIOR, POINT_PRIORS, point_prior    # noqa: E402


def _field(n=40, seed=0):
    """Synthetic residuals with spatial structure on a 200 m x 200 m field."""
    rng = np.random.default_rng(seed); S = rng.uniform(0, 200, (n, 2))
    D = np.linalg.norm(S[:, None] - S[None], axis=-1); C = np.exp(-D / 60) + 1e-9 * np.eye(n)
    r = np.linalg.cholesky(C) @ rng.normal(size=n) + 0.5 * rng.normal(size=n)
    return S, r


def test_settings():
    assert S_.TABPFN_VERSION == "2.0.9" and S_.INFERENCE_CONFIG == {"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,)}
    assert S_.RANDOM_STATE == 0 and S_.N_SPECTRAL_PCS == 10 and S_.BLAS_THREADS == 1 and S_.N_MIXTURE == 400
    assert S_.ZONE_K == 4 and S_.ZONE_MIN_POINTS == 3 and abs(S_.Z80 - 1.2815515655446004) < 1e-15
    assert S_.N_ESTIMATORS == 4 and S_.OOF_FOLDS == 5 and S_.KAPPA == 1.15
    assert S_.POINT_GP_MIN_N == 25 and S_.BLOCK_GP_MIN_N == 3 and S_.BLOCK_MEAN_CORRECTION_MIN_N == 25
    assert list(S_.GRID_LOG_RANGE_OFFSETS) == [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5] and S_.GRID_SPATIAL_SHARES == (0.1, 0.25, 0.5, 0.75, 0.9)
    assert BLOCK_PRIOR == {"mu_r": -2.25, "s_r": 2.14, "mu_s": -0.32, "s_s": 1.82} and set(POINT_PRIORS) == {"SOC", "Clay", "pH"}
    assert point_prior("organic_matter") == POINT_PRIORS["SOC"]


def test_grid_weights_and_total_variance():
    S, r = _field()
    g = ResidualField(BLOCK_PRIOR, field_extent(S)).fit(S, r, block_range_floor(S))
    assert len(g.thetas) == 35 and abs(g.w.sum() - 1) < 1e-12
    tot = [np.exp(t[0]) + np.exp(t[2]) for t in g.thetas]
    assert np.allclose(tot, tot[0], rtol=1e-12)                               # total variance held at the mode


def test_single_point_block_equals_point_posterior():
    """A block of one location must give the same latent mean and variance as the point posterior (same fit)."""
    S, r = _field(); rng = np.random.default_rng(1); Sq = rng.uniform(0, 200, (6, 2))
    g = ResidualField(BLOCK_PRIOR, field_extent(S)).fit(S, r, block_range_floor(S))
    mu, var, vbar = g.predict_points(Sq)
    blk, vbar_b = g.block_moments(Sq, {i: np.array([i]) for i in range(len(Sq))})
    assert abs(vbar - vbar_b) < 1e-12
    for i in range(len(Sq)):
        assert abs(blk[i][0] - mu[i]) < 1e-10 and abs(blk[i][1] - var[i]) < 1e-10


def test_block_variance_does_not_vanish():
    """With correlated errors, the variance of a large block's mean stays well above the independent-error value."""
    S, r = _field(n=60, seed=2); rng = np.random.default_rng(3); Sq = rng.uniform(0, 200, (150, 2))
    g = ResidualField(BLOCK_PRIOR, field_extent(S)).fit(S, r, block_range_floor(S))
    _, var, _ = g.predict_points(Sq); blk, _ = g.block_moments(Sq, {"field": np.arange(150)})
    assert blk["field"][1] > 3 * var.mean() / 150


def test_block_predictive_combination():
    m_T = np.array([1.0, 2.0, 3.0, 4.0]); v_T = np.array([0.5, 0.5, 0.5, 0.5]); blocks = {"field": np.arange(4)}
    out = block_predictive(m_T, v_T, {"field": (0.3, 0.1)}, v_bar=0.2, n_train=10, blocks=blocks)["field"]
    assert abs(out["mean"] - 2.5) < 1e-12                                     # n < 25: no GP mean correction
    assert abs(out["sd"] - S_.KAPPA * np.sqrt(0.1 + 4 * 0.3 / 16)) < 1e-12
    out50 = block_predictive(m_T, v_T, {"field": (0.3, 0.1)}, v_bar=0.2, n_train=50, blocks=blocks)["field"]
    assert abs(out50["mean"] - 2.8) < 1e-12                                   # n >= 25: correction applied


def test_tabpfn_bars_moments_match_quantiles():
    rng = np.random.default_rng(4); borders = np.linspace(-3, 3, 61); logits = rng.normal(size=(3, 60))
    b = TabPFNBars(borders, logits); m, v = b.moments()
    q = b.quantiles((np.arange(200000) + 0.5) / 200000)                      # mean and variance by numerical integration over p
    assert np.allclose(m, q.mean(1), atol=2e-3) and np.allclose(v, q.var(1), rtol=5e-3)


def test_mixture_crps_matches_numerical_integral():
    rng = np.random.default_rng(5); centers = rng.normal(size=(2, 50)); sd = np.array([0.3, 0.7]); y = np.array([0.2, -1.0])
    mix = GaussianMixture(centers, sd); x = np.linspace(-12, 12, 200001)
    for i in range(2):
        F = GaussianMixture(np.repeat(centers[i:i + 1], len(x), 0), np.full(len(x), sd[i])).cdf(x)
        num = np.trapezoid((F - (x >= y[i])) ** 2, x) if hasattr(np, "trapezoid") else np.trapz((F - (x >= y[i])) ** 2, x)
        assert abs(mix.crps(y)[i] - num) < 1e-4


def test_tabpfn_bars_crps_matches_quantile_integral():
    """CRPS = 2 * integral over p of the pinball loss of the p-quantile (checks the exact bar CRPS, tails included)."""
    rng = np.random.default_rng(6); borders = np.linspace(-3, 3, 61); logits = rng.normal(size=(3, 60))
    b = TabPFNBars(borders, logits); y = np.array([0.1, -3.5, 4.2])                 # inside, below and above the bars
    p = (np.arange(400000) + 0.5) / 400000; q = b.quantiles(p)
    num = 2 * np.mean(((y[:, None] < q) - p) * (q - y[:, None]), axis=1)
    assert np.allclose(b.crps(y), num, rtol=1e-3, atol=1e-5)


def test_boolean_mask_blocks_and_block_checks():
    m_T = np.arange(6.0); v_T = np.full(6, 0.4); gp = {"a": (0.2, 0.05)}
    idx = block_predictive(m_T, v_T, gp, 0.1, 60, {"a": np.array([1, 3, 5])})["a"]
    msk = block_predictive(m_T, v_T, gp, 0.1, 60, {"a": np.array([False, True, False, True, False, True])})["a"]
    assert idx["mean"] == msk["mean"] and idx["sd"] == msk["sd"]                    # a mask is read as positions, not as 0/1 indices
    for bad in ({"a": np.array([], int)}, {"a": np.array([0, 6])}, {"a": np.array([True, False])}):
        try:
            as_index_blocks(bad, 6); raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def test_zone_blocks():
    rng = np.random.default_rng(7); S = rng.uniform(0, 300, (80, 2)); te = np.sort(rng.choice(80, 40, replace=False))
    b1, b2 = zone_blocks(S, te), zone_blocks(S, te)
    assert list(b1["field"]) == list(range(40)) and all(np.array_equal(b1[k], b2[k]) for k in b1)   # deterministic
    zones = [v for k, v in b1.items() if k != "field"]
    assert all(len(v) >= S_.ZONE_MIN_POINTS for v in zones) and len(np.concatenate(zones)) == len(np.unique(np.concatenate(zones)))


def test_input_checks():
    for bad in (np.array([[0.0, 0.0], [np.nan, 1.0], [2.0, 3.0]]), np.zeros((4, 3))):
        try:
            field_extent(bad); raise AssertionError("field_extent accepted bad coordinates")
        except ValueError:
            pass
    try:
        CoordinateTransform("utm"); CoordinateTransform("utm").fit(np.zeros((3, 2))); raise AssertionError("accepted crs='utm'")
    except ValueError:
        pass
    from tabpfn_spatialgp import TabPFNSpatialGP                                    # these checks run before TabPFN is used
    X, S, y = np.zeros((60, 2)), np.random.default_rng(8).uniform(0, 100, (60, 2)), np.linspace(0, 1, 60)
    for args, kw in (((X[:5], S, y), {}), ((X[:2], S[:2], y[:2]), {}), ((X, S, np.ones(60)), {}), ((X, S, y), {})):
        try:
            TabPFNSpatialGP(**kw).fit(*args, field_coords=S); raise AssertionError("fit accepted bad input")
        except ValueError:                                                          # the last case: n >= 50 without target_kind
            pass
