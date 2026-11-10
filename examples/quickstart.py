"""Quickstart: field and zone means of a soil property, with 80 % intervals, from 15 samples.

    python examples/quickstart.py

A synthetic field of 200 sampled locations has two covariates known everywhere (think of a conductivity survey and elevation) and
a soil property that depends on them plus a spatially correlated part. Fifteen locations are "sent to the laboratory"; TabPFN-GP
predicts the mean of the property over the other 185 locations and over four compact zones, and the script compares each 80 % interval
with the true mean. TabPFN runs on the GPU if there is one, otherwise on the CPU (slower). The first run downloads TabPFN's weights.
"""
import numpy as np
import pandas as pd

from tabpfn_spatialgp import TabPFNGPRegressor, zone_blocks


def synthetic_field(n=200, seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 300, (n, 2))                                            # projected coordinates, metres
    ec, elev = rng.normal(size=n), rng.normal(size=n)                           # covariates known at every location
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    spatial = np.linalg.cholesky(np.exp(-d / 80) + 1e-8 * np.eye(n)) @ rng.normal(size=n)
    soc = 2.0 + 0.6 * ec - 0.3 * elev + 0.4 * spatial + 0.1 * rng.normal(size=n)   # e.g. soil organic carbon, %
    return pd.DataFrame({"ec": ec, "elevation": elev, "x": xy[:, 0], "y": xy[:, 1]}), soc


def main():
    field, soc = synthetic_field()
    rng = np.random.default_rng(0)
    sampled = rng.choice(len(field), 15, replace=False)                         # the laboratory samples
    rest = np.setdiff1d(np.arange(len(field)), sampled)                         # where the means are wanted

    reg = TabPFNGPRegressor(coord_cols=["x", "y"], target_kind="SOC")
    reg.fit(field.iloc[sampled], soc[sampled], field_coords=field[["x", "y"]].to_numpy())

    blocks = zone_blocks(field[["x", "y"]].to_numpy(), rest)                    # the whole field and four compact zones
    means = reg.predict_blocks(field.iloc[rest], blocks)

    print(f"{'area':8s} {'locations':>9s} {'predicted':>9s} {'80 % interval':>17s} {'true mean':>9s}  covered")
    for name, out in means.items():
        lo, hi = out["interval80"]; true = soc[rest[blocks[name]]].mean()
        print(f"{name:8s} {len(blocks[name]):9d} {out['mean']:9.3f} [{lo:6.3f}, {hi:6.3f}] {true:9.3f}  {'yes' if lo <= true <= hi else 'no'}")
    print(f"\nsample mean of the 15 samples: {soc[sampled].mean():.3f}; true field mean: {soc[rest].mean():.3f}")


if __name__ == "__main__":
    main()
