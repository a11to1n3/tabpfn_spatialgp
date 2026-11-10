"""Development check (docs/development/plan_dev_checks.md): independent part from the GP's nugget instead of TabPFN's variance.

  rows:   python -m experiments.development.dev_nugget rows OUT [workers]
  report: python -m experiments.development.dev_nugget report OUT
CPU only, from the stored development point outputs (SOILGP_DATA, SOILGP_RUN as in the development run).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

MODE, OUT = sys.argv[1], Path(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.development.ablation import POP                              # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent   # noqa: E402

VARIANTS = {"base": "the method", "gp_nugget": "GP's nugget instead of TabPFN's variance", "no_split": "all of v_T as independent"}
DEV = list(F.C.DEV_FIELDS)


def dev_cases(field):                                                         # as in development.py
    for f, data, design, rep, tr in block_cases.iter_splits("dev", [field], min_test=1):
        for target in F.targets(f, "dev"):
            if not np.all(np.isfinite(data["Y"][:, F.C.TARGETS.index(target)])):
                continue
            case = block_cases.load_case("dev", f, data, design, rep, tr, target)
            if case is not None:
                yield case


def rows_field(field):
    t0 = time.monotonic(); rows = []
    for case in dev_cases(field):
        S, tr, te, y = case["S"], case["tr"], case["te"], case["y"]; n = len(tr); yq = y[te]
        blocks, revealed = block_cases.blocks_of(case); mT, sT = case["mT"], case["sT"]; gate = n >= F.GATE_N

        def add(v, b, M, V, err=""):
            i = blocks[b]; failed = bool(err) or not (np.isfinite(M) and np.isfinite(V) and V > 1e-18)
            rows.append(dict(set="dev", src="limesoda", field=field, design=case["design"], rep=case["rep"], target=case["target"], n=n,
                             m=len(te), block=b, mB=len(i), nB=len(revealed[b]), variant=v, M=float(M), V=float(V), df=np.inf,
                             obs=float(yq[i].mean()), failed=failed, err=err))
        try:
            g = ResidualField(POP, field_extent(S)).fit(S[tr], y[tr] - case["m_oof"], block_range_floor(S[tr]))
            moments, vbar = g.block_moments(S[te], blocks)
            nug = g.ys ** 2 * float(np.dot(g.w, [g._unpack(th)["tau"] for th in g.thetas]))   # nugget variance, mixed over the grid
        except Exception as e:
            for v in VARIANTS:
                for b in blocks:
                    add(v, b, np.nan, np.nan, err=repr(e)[:150])
            continue
        ind = np.clip(sT ** 2 - vbar, 0, None)
        for b, i in blocks.items():
            mb, vgb = moments[b]; m = len(i); M = float(mT[i].mean()) + (mb if gate else 0.0)
            add("base", b, M, vgb + float(np.sum(ind[i])) / m ** 2)
            add("gp_nugget", b, M, vgb + nug / m)
            add("no_split", b, M, vgb + float(np.sum(sT[i] ** 2)) / m ** 2)
    pd.DataFrame(rows).to_csv(OUT / f"nugget_dev_{field}.csv", index=False)
    return field, len(rows), time.monotonic() - t0


def report():
    rows = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("nugget_dev_*.csv"))], ignore_index=True)
    rows["failed"] = rows.failed.astype(bool)
    print(f"blocks per variant {int((rows.variant == 'base').sum())}; failures {int(rows[rows.variant == 'base'].failed.sum())}; "
          f"fields {rows.field.nunique()}")

    def block(sub, k, lab):
        r = scoring.score(sub, {v: k for v in VARIANTS}); common = scoring.common_blocks(r, list(VARIANTS))
        cov = {v: [float(scoring.kind_rows(common[common.variant == v], kd).cov80.mean()) for kd in ("field", "zone")] for v in VARIANTS}
        for alt in ("gp_nugget", "no_split"):
            line(common, cov, alt, lab)

    def line(common, cov, alt, lab):
        c = [scoring.contrast(common, alt, "base", kd) for kd in ("field", "zone")]
        txt = []
        for j, kd in enumerate(("field", "zone")):
            txt.append("-" if c[j] is None else f"{c[j]['ratio']:.3f} [{c[j]['ci'][0]:.3f}, {c[j]['ci'][1]:.3f}] ({c[j]['better']}/{c[j]['fields']} fields lower)")
        better = all(x is not None and x["ratio"] < 1 for x in c) and all(abs(cov[alt][j] - 0.8) <= abs(cov["base"][j] - 0.8) for j in range(2))
        print(f"  {lab}: coverage base {cov['base'][0]:.3f} | {cov['base'][1]:.3f}; {alt} {cov[alt][0]:.3f} | {cov[alt][1]:.3f}; "
              f"CRPS {alt} / base {txt[0]} | {txt[1]}" + ("  (better on the rule)" if better else ""))

    for k in (1.0, 1.10):
        print(f"\nkappa = {k:.2f} (field | zone)")
        block(rows, k, "all")
        for lo, hi in ((5, 10), (15, 20), (30, 10 ** 6)):
            block(rows[rows.n.between(lo, hi)], k, f"n {lo}-{hi if hi < 10 ** 6 else 'max'}")
        z = rows.block != "field"
        for lo, hi in ((3, 9), (10, 29), (30, 10 ** 6)):
            sub = rows[(~z) | rows.mB.between(lo, hi)]
            block(sub, k, f"zones m_B {lo}-{hi if hi < 10 ** 6 else 'max'} (field column = all field means)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if MODE == "rows":
        with Pool(WORKERS) as pool:
            for field, k, s in pool.imap_unordered(rows_field, DEV):
                print(f"rows {field}: {k} rows, {s:.0f} s", flush=True)
    else:
        report()
