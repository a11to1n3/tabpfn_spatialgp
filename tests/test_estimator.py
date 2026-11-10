"""The scikit-learn interface TabPFNGPRegressor.
  test_estimator_params_and_clone    get_params, set_params and clone, without TabPFN.
  test_estimator_columns             coordinate, covariate and spectral columns by name and by position, and their checks, without TabPFN.
  test_estimator_matches_core (GPU, only with RUN_GPU=1)   predict, quantiles and field and zone means equal TabPFNSpatialGP's exactly.
  test_estimator_in_sklearn (GPU, only with RUN_GPU=1)     clone, cross_val_score, a Pipeline and pickling work.
  test_estimator_sklearn_checks (GPU, only with RUN_GPU=1) scikit-learn's estimator checks pass, except the two expected ones:
      check_estimators_pickle         puts NaN in random cells of X, coordinates included; a location without coordinates cannot be
                                      placed (missing covariates are allowed, and pickling itself works: test_estimator_in_sklearn);
      check_methods_subset_invariance TabPFN's predictions change slightly with the set of rows predicted together (about 2e-3 on a
                                      property with standard deviation 1, from TabPFN 2.0.9 itself).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabpfn_spatialgp import TabPFNGPRegressor, TabPFNSpatialGP, zone_blocks           # noqa: E402


def _field(n=160, seed=3):
    """A synthetic field: two covariates and coordinates in metres, a property with a trend in the covariates and a spatial part."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(0, 300, (n, 2)); X = rng.normal(size=(n, 2))
    D = np.linalg.norm(S[:, None] - S[None], axis=-1); C = np.exp(-D / 80) + 1e-8 * np.eye(n)
    y = 2.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + np.linalg.cholesky(C) @ rng.normal(size=n) * 0.4 + 0.1 * rng.normal(size=n)
    return X, S, y


def test_estimator_params_and_clone():
    from sklearn.base import clone
    reg = TabPFNGPRegressor(coord_cols=("x", "y"), target_kind="SOC", kappa=1.25)
    params = reg.get_params()
    assert params["coord_cols"] == ("x", "y") and params["target_kind"] == "SOC" and params["kappa"] == 1.25
    assert set(params) == {"coord_cols", "spectral_cols", "target_kind", "point_prior", "kappa", "crs", "device", "model_path"}
    c = clone(reg); assert c is not reg and c.get_params() == params
    assert reg.set_params(kappa=1.15).kappa == 1.15


def test_estimator_columns():
    import pandas as pd
    X, S, y = _field(20)
    df = pd.DataFrame({"ec": X[:, 0], "wl_500": X[:, 1], "x": S[:, 0], "y": S[:, 1]})
    reg = TabPFNGPRegressor(coord_cols=("x", "y"), spectral_cols=["wl_500"])
    Xc, coords = reg._split(df, fitting=True)
    assert np.array_equal(coords, S) and np.array_equal(Xc, X) and list(reg.spectral_mask_) == [False, True]
    assert list(reg.feature_names_in_) == ["ec", "wl_500", "x", "y"] and reg.n_features_in_ == 4
    Xc, coords = TabPFNGPRegressor()._split(np.column_stack([X, S]), fitting=True)        # default: the last two columns
    assert np.array_equal(coords, S) and np.array_equal(Xc, X)
    for kw, data in ((dict(coord_cols=("x", "y")), np.column_stack([X, S])),            # names with an array
                     (dict(coord_cols=("x", "z")), df), (dict(coord_cols=("x",)), df), (dict(coord_cols=("x", "x")), df),
                     (dict(coord_cols=(0, 9)), np.column_stack([X, S])), (dict(spectral_cols=["x"], coord_cols=("x", "y")), df)):
        try:
            TabPFNGPRegressor(**kw)._split(data, fitting=True); raise AssertionError(f"accepted {kw}")
        except ValueError:
            pass
    try:
        TabPFNGPRegressor().predict(np.column_stack([X, S])); raise AssertionError("predict before fit")
    except Exception as e:                                                         # sklearn's NotFittedError
        assert type(e).__name__ == "NotFittedError", type(e)


def _gpu():
    if os.environ.get("RUN_GPU") != "1":
        raise unittest.SkipTest("set RUN_GPU=1 to fit TabPFN on the GPU")


def test_estimator_matches_core():
    _gpu()
    X, S, y = _field()
    rng = np.random.default_rng(0)
    for n in (10, 40):                                                             # below and above the 25-sample threshold
        tr = rng.choice(len(y), n, replace=False); te = np.setdiff1d(np.arange(len(y)), tr)
        core = TabPFNSpatialGP(target_kind="SOC").fit(X[tr], S[tr], y[tr], field_coords=S)
        reg = TabPFNGPRegressor(target_kind="SOC").fit(np.column_stack([X[tr], S[tr]]), y[tr], field_coords=S)
        Xq = np.column_stack([X[te], S[te]])
        a, b = core.predict(X[te], S[te]), reg.predict(Xq)
        assert np.array_equal(a["mean"], b), n
        assert np.array_equal(np.column_stack([a["quantiles"][q] for q in (0.1, 0.5, 0.9)]), reg.predict_quantiles(Xq)), n
        blocks = zone_blocks(S, te)
        ca, cb = core.predict_blocks(X[te], S[te], blocks), reg.predict_blocks(Xq, blocks)
        assert set(ca) == set(cb) and all(ca[k]["mean"] == cb[k]["mean"] and ca[k]["sd"] == cb[k]["sd"] for k in ca), n
        print(f"    n = {n}: predictions, quantiles and {len(cb)} field and zone means identical")


def test_estimator_in_sklearn():
    _gpu()
    from sklearn.base import clone
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    X, S, y = _field(60)
    XS = np.column_stack([X, S])
    scores = cross_val_score(TabPFNGPRegressor(target_kind="SOC"), XS, y, cv=KFold(3, shuffle=True, random_state=0))
    assert np.all(np.isfinite(scores)), scores
    pipe = Pipeline([("tabpfn_gp", TabPFNGPRegressor(target_kind="SOC"))]).fit(XS[:40], y[:40])
    assert np.all(np.isfinite(pipe.predict(XS[40:])))
    assert clone(pipe).get_params()["tabpfn_gp__target_kind"] == "SOC"
    import pickle
    reg = pickle.loads(pickle.dumps(pipe.named_steps["tabpfn_gp"]))                  # a fitted estimator survives pickling
    assert np.array_equal(reg.predict(XS[40:]), pipe.predict(XS[40:]))
    print(f"    cross_val_score R^2: {np.round(scores, 3)}")


EXPECTED_FAILURES = {"check_estimators_pickle", "check_methods_subset_invariance"}


def test_estimator_sklearn_checks():
    _gpu()
    import warnings
    from sklearn.utils import estimator_checks as ec
    est = TabPFNGPRegressor(crs="auto", target_kind="SOC")                      # crs='auto': the checks' coordinates are small numbers
    gen = (ec.estimator_checks_generator(est, legacy=True) if hasattr(ec, "estimator_checks_generator")
           else ec.check_estimator(est, generate_only=True))
    failed, n = set(), 0
    for e, check in gen:
        n += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                check(e)
            except Exception:
                failed.add(getattr(check, "func", check).__name__)
    assert failed <= EXPECTED_FAILURES, sorted(failed - EXPECTED_FAILURES)
    print(f"    {n} scikit-learn checks; failed only the expected ones: {sorted(failed)}")
