"""Build the field files <SOILGP_DATA>/<field>.npz used by every experiment.

Each file holds X (covariates), Y (n x 3: SOC, clay, pH; organic matter in the SOC slot where SOC is not measured), columns, targets,
folds (LimeSoDa's 10 folds; 0 where a dataset has none) and coordinates (projected metres).

Sources:
  - 18 LimeSoDa datasets with coordinates, via the LimeSoDa Python package (--limesoda: its src folder; CC BY-SA 4.0). Spectral columns
    must start with 'wl_'; MIR wavenumber columns 'wn_...' are therefore renamed 'wl_wn_...'. SL.125 and UL.120 have dummy WGS84
    coordinates in LimeSoDa, given for privacy; they are projected to metres with a local equirectangular projection around their mean
    (sphere of radius 6,371,008.8 m, offsets 500,000 / 6,000,000 m).
  - The SensRes fields LYD.112 (Lyduokiai) and RUM.70 (Rumokai), Lithuania (SensRes dataset, Zenodo doi:10.5281/zenodo.14161809,
    CC BY 4.0), via export_sensres.R (--sensres: the unzipped SensRes_Lithuania folder). Covariates are read at the sample points;
    coordinates are EPSG:3346 (metres), as released.
  - Oxford (gstat::oxford) and Meuse (sp::meuse), via export_external.R. OXF: organic matter, clay, pH; covariates elevation, soil-map
    class and topsoil colour. OXFNC: the same without colour. MEUSE: organic matter in all three slots (only the SOC slot is analysed).

Usage: python -m experiments.data.export_data --limesoda <LimeSoDa>/src --sensres <SensRes_Lithuania> [--check <folder with field files to compare with>]
--check compares every array with the reference files by value; the files themselves can differ in bytes (dtypes, memory order,
zip layout), so the byte fingerprint of checks.py identifies only the reference field files.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.core import datasets as C

EARTH_RADIUS = 6371008.8
DUMMY_WGS84 = ("SL.125", "UL.120")
SENSRES_CODES = {"LYD.112": "LT_LYD", "RUM.70": "LT_RUM"}


def project_dummy_wgs84(lonlat):
    lon, lat = np.asarray(lonlat, float).T; lon0, lat0 = lon.mean(), lat.mean()
    x = 500000 + EARTH_RADIUS * np.cos(np.radians(lat0)) * np.radians(lon - lon0)
    y = 6000000 + EARTH_RADIUS * np.radians(lat - lat0)
    return np.column_stack([x, y])


def limesoda_field(field, load_dataset):
    d = load_dataset(field); frame = d["Dataset"]
    cols = [c for c in frame.columns if not c.endswith("_target")]
    soc = "SOC_target" if "SOC_target" in frame else "SOM_target"
    arrays = {"X": frame[cols].to_numpy(float), "Y": frame[[soc, "Clay_target", "pH_target"]].to_numpy(float),
              "columns": np.array(["wl_" + c if str(c).startswith("wn_") else c for c in cols]), "targets": np.array(C.TARGETS),
              "folds": np.asarray(d["Folds"], int)}
    S = np.asarray(d["Coordinates"], float)
    if field in DUMMY_WGS84:
        arrays["coordinates_wgs84"] = S; S = project_dummy_wgs84(S)
    arrays["coordinates"] = S
    assert all(np.isfinite(arrays[k]).all() for k in ("X", "Y", "coordinates")), field
    return arrays


def sensres_fields(folder):
    """LYD.112 and RUM.70: targets SOC (%), clay (%) and pH as released; covariates ECa, Sentinel-2 bands and UAV RGB at the points."""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["Rscript", str(Path(__file__).with_name("export_sensres.R")), str(Path(folder).expanduser()), tmp], check=True)
        for field, code in SENSRES_CODES.items():
            t = pd.read_csv(Path(tmp) / f"{code}_points.csv")
            cols = [c for c in t.columns if c not in ("ID", "x", "y", "SOC", "clay", "pH")]
            S = t[["x", "y"]].to_numpy(float)
            assert len(t) == int(field.split(".")[1]) and len(np.unique(S, axis=0)) == len(S), field
            out[field] = {"X": t[cols].to_numpy(float), "Y": t[["SOC", "clay", "pH"]].to_numpy(float), "columns": np.array(cols),
                          "targets": np.array(C.TARGETS), "folds": np.zeros(len(t), np.int64), "coordinates": S}
    return out


def external_fields():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["Rscript", str(Path(__file__).with_name("export_external.R")), tmp], check=True)
        ox, mz = pd.read_csv(Path(tmp) / "oxford.csv"), pd.read_csv(Path(tmp) / "meuse.csv")
    def build(X, Y, S):
        return {"X": np.asarray(X, float), "Y": np.asarray(Y, float), "columns": np.array(list(X.columns)), "targets": np.array(C.TARGETS),
                "folds": np.zeros(len(S), int), "coordinates": np.asarray(S, float)}
    return {"OXF": build(ox[["ELEV", "MAPCLASS", "VAL1", "CHR1"]], ox[["OM", "CLAY", "PH"]], ox[["x", "y"]]),
            "OXFNC": build(ox[["ELEV", "MAPCLASS"]], ox[["OM", "CLAY", "PH"]], ox[["x", "y"]]),
            "MEUSE": build(mz[["elev", "dist", "ffreq", "soil"]], np.column_stack([mz.OM] * 3), mz[["x", "y"]])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limesoda", required=True); ap.add_argument("--sensres", required=True)
    ap.add_argument("--check", default=None, help="folder with field files to compare with"); a = ap.parse_args()
    sys.path.insert(0, str(Path(a.limesoda).expanduser())); from LimeSoDa import load_dataset
    fields = {f: limesoda_field(f, load_dataset) for f in C.LIMESODA if f not in C.SENSRES}
    fields.update(sensres_fields(a.sensres))
    fields.update(external_fields())
    C.DATA.mkdir(parents=True, exist_ok=True); bad = 0
    for f, arrays in fields.items():
        np.savez(C.DATA / f"{f}.npz", **arrays)
        if a.check:                                   # every array of the reference file, compared by value (dtype and layout aside)
            ref = np.load(Path(a.check).expanduser() / f"{f}.npz", allow_pickle=True)
            same = {k: k in arrays and bool(np.array_equal(np.asarray(arrays[k]), ref[k])) for k in ref.files}
            same["no extra arrays"] = set(arrays) <= set(ref.files)
            bad += not all(same.values()); print(f, "identical in every array" if all(same.values()) else f"DIFFERS {same}")
    print(f"{len(fields)} fields written to {C.DATA}" + (f"; {bad} differ from {a.check}" if a.check else ""))
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
