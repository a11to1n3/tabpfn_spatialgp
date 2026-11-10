"""Checks of the field files and of the package. Any failure stops the run.

  python -m experiments.data.checks fingerprint
      (a) the field files: the sha256 of the reference files, or, for files rebuilt with export_data.py (equal in value, not in
      bytes), the sha256 of their arrays
  python -m experiments.data.checks equivalence
      (b) GPU: the package's fit + predict_blocks against the experiment code's TabPFN-GP on 67 cases (n = 10, replicate 120, every
      field and property), relative tolerance 1e-10
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import hashlib                                                                # noqa: E402

import numpy as np                                                            # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402

# sha256 of the lines "<sha256 of file>  <file name>" in Python sort order (the shell `sha256sum *.npz | sha256sum` gives a8fb267d...
# for the same per-file hashes, because it sorts by locale)
DATA_FINGERPRINT = "304f00243e4c47d4f5df1bba0d0f9a198d04145108d48104f615a30c7878f4b2"   # the development and factor-setting field files (18 LimeSoDa fields, LYD.112, RUM.70, OXF, MEUSE, OXFNC)
DATA_VALUE_FINGERPRINT = "ca33f9651f378e3cea3449735c2d09b5a1524c6808aaa0464f87c3eddd7b7197"   # value_fingerprint() of the same files


def value_fingerprint(files):
    """sha256 over the arrays of every field file, by value: numeric arrays as float64 or int64 in C order, the others as text."""
    h = hashlib.sha256()
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            for k in sorted(z.files):
                a = np.asarray(z[k])
                if a.dtype.kind in "fc":
                    b = np.ascontiguousarray(a, dtype=np.float64).tobytes()
                elif a.dtype.kind in "iub":
                    b = np.ascontiguousarray(a, dtype=np.int64).tobytes()
                else:
                    b = "\x1f".join(map(str, a.ravel().tolist())).encode()
                h.update(f"{f.name}|{k}|{a.shape}|".encode()); h.update(b)
    return h.hexdigest()


def fingerprint():
    files = sorted(F.C.DATA.glob("*.npz"))
    lines = "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in files)
    got = hashlib.sha256(lines.encode()).hexdigest()
    print(f"{len(files)} files, fingerprint {got}")
    assert len(files) == 23, "expected 23 field files"
    if got != DATA_FINGERPRINT:
        val = value_fingerprint(files)
        print(f"bytes differ from the reference files; fingerprint of the values {val}")
        assert val == DATA_VALUE_FINGERPRINT, "data values differ from the reference files"
    print("(a) data fingerprint: OK")


def equivalence():
    from experiments.core.block_model import PRIORS, block_model              # noqa: F401  (the experiment code's TabPFN-GP)
    from tabpfn_spatialgp import TabPFNSpatialGP
    from tabpfn_spatialgp.blocks import zone_blocks
    from tabpfn_spatialgp.preprocess import spectral_mask_from_columns
    worst, n_cases = 0.0, 0
    for field, data, design, rep, tr in block_cases.iter_splits("final"):
        if design != "n10" or rep != 120:
            continue
        S = np.asarray(data["coordinates"], float); te = np.setdiff1d(np.arange(len(S)), tr)
        for target in F.targets(field, "final"):
            y = data["Y"][:, F.C.TARGETS.index(target)]
            case = block_cases.load_case("final", field, data, design, rep, tr, target)
            assert case is not None, (field, target)
            exp_rows = {r["block"]: r for r in block_cases.case_rows(case, None, ours=True) if r["variant"] == "J1pK"}
            m = TabPFNSpatialGP().fit(data["X"][tr], S[tr], y[tr], field_coords=S, spectral_mask=spectral_mask_from_columns(data["columns"]))
            pkg = m.predict_blocks(data["X"][te], S[te], zone_blocks(S, te), kappa=F.KAPPA_OURS)   # the evaluated 1.10, not the default
            for b, r in exp_rows.items():
                dm = abs(pkg[b]["mean"] - r["M"]) / max(abs(r["M"]), 1e-12)
                dv = abs(pkg[b]["sd"] ** 2 - r["V"]) / max(r["V"], 1e-18)
                worst = max(worst, dm, dv)
            n_cases += 1
    print(f"{n_cases} cases; largest relative difference {worst:.2e}")
    assert n_cases == 67 and worst < 1e-10, "package equivalence failed"
    print("(b) package equivalence: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("check", choices=["fingerprint", "equivalence"]); a = ap.parse_args()
    {"fingerprint": fingerprint, "equivalence": equivalence}[a.check]()
