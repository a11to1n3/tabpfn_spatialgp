"""Shared pieces of the experiments: data, training sets, TabPFN, predictive distributions, the R bridge (KpR), output files."""
import json
import re
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_spatialgp import settings as S_
from tabpfn_spatialgp.bars import TabPFNBars
from tabpfn_spatialgp.preprocess import CovariateTransform, spectral_mask_from_columns

from experiments.core import datasets as C

KRIGING_R = Path(__file__).resolve().parent / "kriging.R"


# ----------------------------------------------------------------------------------------------------------------------------- data
def load_field(field):
    """X (covariates), Y (n x 3: SOC or organic matter, clay, pH), coordinates (projected metres), columns, folds (KpR paper; 0 if none)."""
    d = np.load(C.DATA / f"{field}.npz", allow_pickle=True)
    return {k: d[k] for k in ("X", "Y", "coordinates", "columns", "folds")}


def training_rows(data, field, design, rep, min_test=10):
    """Training rows of one split. Every design is seeded by crc32('field|design|rep'), so all methods see the same split.
      n<N>       N random samples (ValueError if fewer than min_test samples would be left as test points)
      random<P>  P % random samples
      cluster<P> P % in 4 compact clusters: 4 random seed samples, then round-robin nearest-neighbour growth
      fold<k>    the KpR authors' published 10-fold CV: all folds except k"""
    S = np.asarray(data["coordinates"], float); n = len(S)
    if design.startswith("fold"):
        return np.flatnonzero(np.asarray(data["folds"]) != int(design[4:]))
    rng = np.random.default_rng(zlib.crc32(f"{field}|{design}|{rep}".encode()))
    if re.fullmatch(r"n\d+", design):
        N = int(design[1:])
        if N > n - min_test:
            raise ValueError(f"{design} infeasible for {field} ({n} samples)")
        return np.sort(rng.choice(n, N, replace=False))
    kind, pct = re.fullmatch(r"(random|cluster)(\d+)", design).groups(); m = int(round(int(pct) / 100 * n))
    if kind == "random":
        return np.sort(rng.choice(n, m, replace=False))
    seeds = rng.choice(n, 4, replace=False); chosen = set(seeds.tolist())
    order = [np.argsort(np.linalg.norm(S - S[s], axis=1)) for s in seeds]; ptr = [0] * 4; j = 0
    while len(chosen) < m:
        o = order[j % 4]
        while o[ptr[j % 4]] in chosen:
            ptr[j % 4] += 1
        chosen.add(int(o[ptr[j % 4]])); j += 1
    return np.array(sorted(chosen))


def covariates(data, tr, te, with_coords):
    """Covariates for TabPFN: spectra -> principal components fitted on the training rows; coordinates appended if with_coords."""
    t = CovariateTransform(spectral_mask_from_columns(data["columns"])).fit(data["X"][tr])
    xa, xb = t.transform(data["X"][tr]), t.transform(data["X"][te])
    if with_coords:
        S = np.asarray(data["coordinates"], float); xa, xb = np.column_stack([xa, S[tr]]), np.column_stack([xb, S[te]])
    return xa, xb


# ------------------------------------------------------------------------------------------------------------------------- backbones
_PINNED = False


def single_thread():
    """TabPFN's output depends on the number of BLAS/OpenMP threads for some inputs; the study's runs used one thread. torch and
    tabpfn are loaded first, because threadpoolctl only limits libraries that are already loaded."""
    global _PINNED
    if not _PINNED:
        import tabpfn  # noqa: F401
        import torch
        from threadpoolctl import threadpool_limits
        threadpool_limits(1); torch.set_num_threads(1); _PINNED = True


def tabpfn(n_estimators=S_.N_ESTIMATORS, target_transform=False, seed=S_.RANDOM_STATE):
    """TabPFN v2.0.9 with its original weights. Ours: 4 members, no target transform (= tabpfn_spatialgp.backbone.make_regressor).
    TabPFN's defaults (8 members, its default target transform) are used for KpR and TabPFN-RK, as in the KpR paper."""
    single_thread()
    from tabpfn import TabPFNRegressor
    kw = {} if target_transform else {"inference_config": S_.INFERENCE_CONFIG}
    return TabPFNRegressor(device="cuda", n_estimators=n_estimators, random_state=seed, **kw)


def default_tabpfn():
    return tabpfn(n_estimators=8, target_transform=True)


def bars(regressor, x):
    """TabPFN's predictive distribution at rows x."""
    return TabPFNBars.from_tabpfn(regressor, x)


def bar_output(dist, y):
    """mean, CRPS and quantiles of a predictive distribution (TabPFNBars or GaussianMixture) at observations y."""
    return {"mean": dist.moments()[0], "crps": dist.crps(y), "quantiles": dist.quantiles(np.array(C.QUANTILES))}


def interval_sd(quantiles):
    """Gaussian-equivalent sd of the central 80 % interval: (Q(0.9) - Q(0.1)) / (2 z_0.9)."""
    q = np.asarray(quantiles)
    return (q[:, 9] - q[:, 1]) / (2 * S_.Z80)


# ------------------------------------------------------------------------------------------------------------------------- R bridge
def kriging(mode, S, train_rows, query_rows, values):
    """Run kriging.R (automap/gstat) in a temporary folder and return its outputs: {name: data frame (row, mean, variance)} and 'status'.
    mode 'kpr': values = y at the training rows -> 'kpr_training' (leave-one-out kriging), 'kpr_query' (KpR features)"""
    work = Path(tempfile.mkdtemp(prefix="krige_"))
    try:
        S = np.asarray(S, float); cols = {"row": train_rows, "s_x": S[train_rows, 0], "s_y": S[train_rows, 1]}
        cols.update(values if isinstance(values, dict) else {"value": values})
        pd.DataFrame(cols).to_csv(work / "training.csv", index=False)
        pd.DataFrame({"row": query_rows, "s_x": S[query_rows, 0], "s_y": S[query_rows, 1]}).to_csv(work / "query.csv", index=False)
        subprocess.run(["Rscript", str(KRIGING_R), mode, str(work)], check=True, capture_output=True, timeout=900)
        status = json.loads((work / "status.json").read_text())
        out = {"status": status}
        for f in work.glob("*_out.csv"):
            out[f.stem[:-4]] = pd.read_csv(f)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------------- files
def point_file(root, model, design, field, target, rep):
    """Point outputs: <root>/<model>/<design>/<field>/<target>/rep<r>.npz."""
    return Path(root) / model / design / field / target / f"rep{rep}.npz"


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays); tmp.replace(path)
