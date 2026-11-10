"""Exploratory component ablation of TabPFN-GP's block model.

Reads the saved point outputs of a set (block_cases.load_case); CPU only, no TabPFN refits. Every variant keeps kappa = 1.10 and changes one
component of the chosen model (J1pK):
  J1pK  the chosen model (checked against the stored rows)
  A0    no GP: TabPFN's estimate, errors independent, variance (1/m^2) sum v_T
  A1    no prior: prior density removed (s_r = s_s = inf); starting values and range bounds unchanged
  A2    no averaging: posterior mode only
  A3    no prior and no averaging
  A4    no offset term: the GLS-mean term u u' / (1' K^-1 1) of the GP block variance removed
  A5    no independent part: variance V_gB only
  A6    in-sample residuals: the GP fitted to y - m_T of the final TabPFN instead of the cross-fitted residuals
  A7    no mean correction at any n
  A8    mean correction at every n

  python -m experiments.development.ablation rows --set first|final --out DIR [--fields ...] [--workers 16] [--designs ...] [--reps ...]
  python -m experiments.development.ablation report --set first|final --out DIR --stored DIR_WITH_rows_<set>_<field>.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from scipy.linalg import cho_solve                                            # noqa: E402
from scipy.spatial.distance import cdist                                      # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.core.block_model import PRIORS                               # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent, matern32   # noqa: E402

POP = PRIORS["POP"]
FLAT = dict(POP, s_r=np.inf, s_s=np.inf)
K2 = F.KAPPA_OURS ** 2
VARIANTS = ("J1pK", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")
LABELS = {"J1pK": "chosen model", "A0": "no GP", "A1": "no prior", "A2": "no averaging", "A3": "no prior, no averaging",
          "A4": "no offset term", "A5": "no independent part", "A6": "in-sample residuals", "A7": "no mean correction",
          "A8": "mean correction at every n"}
FILTER = {}                                                                   # --designs / --reps: for tests only


def offset_terms(g, Sq, blocks):
    """Grid-weighted GLS-mean term of the block variance, as in ResidualField.block_moments (its block mean is mean(u)^2 / A), in the
    original units."""
    Dq = cdist(Sq, g.S); out = dict.fromkeys(blocks, 0.0)
    for th, w in zip(g.thetas, g.w):
        p = g._unpack(th); _, Fc, _, _ = g._nlp(th, ret=True)
        Kikq = cho_solve(Fc, (p["s1"] * matern32(Dq, p["l1"])).T)
        A = float((g.H.T @ cho_solve(Fc, g.H)).ravel()[0]); u = 1.0 - (g.H.T @ Kikq).ravel()
        for b, i in blocks.items():
            out[b] += w * float(u[i].mean()) ** 2 / A
    return {b: g.ys ** 2 * v for b, v in out.items()}


def fit(prior, S, tr, te, r, blocks):
    """Averaged and plug-in GP block moments, and the offset terms of the averaged fit."""
    g = ResidualField(prior, field_extent(S)).fit(S[tr], r, block_range_floor(S[tr]))
    avg = g.block_moments(S[te], blocks); off = offset_terms(g, S[te], blocks)
    g.thetas, g.w = [g.map_theta], np.array([1.0])
    return avg, g.block_moments(S[te], blocks), off


def case_rows(case):
    S, tr, te, y = case["S"], case["tr"], case["te"], case["y"]; n = len(tr); yq = y[te]
    blocks, revealed = block_cases.blocks_of(case)
    mT, sT = case["mT"], case["sT"]; gate = n >= F.GATE_N
    rows = []

    def add(variant, b, M, V, err="", **extra):
        i = blocks[b]; failed = bool(err) or not (np.isfinite(M) and np.isfinite(V) and V > 1e-18)
        rows.append(dict(set=case["set"], src="external" if case["field"] in F.C.EXTERNAL else "limesoda", field=case["field"],
                         design=case["design"], rep=case["rep"], target=case["target"], n=n, m=len(te), block=b, mB=len(i),
                         nB=len(revealed[b]), variant=variant, M=float(M), V=float(V), df=np.inf, obs=float(yq[i].mean()), failed=failed,
                         err=err or ("non-finite mean or variance <= 1e-18" if failed else ""), **extra))

    def assemble(variant, moments, vbar, correct, off=None, independent=True, diag=None):
        ind = np.clip(sT ** 2 - vbar, 0, None)
        for b, i in blocks.items():
            mb, vb = moments[b]; m = len(i)
            vi = float(np.sum(ind[i])) / m ** 2 if independent else 0.0
            vg = vb - (off[b] if off is not None else 0.0)
            extra = {} if diag is None else {"offset_share": diag[b] / (vg + vi), "indep_share": vi / (vg + vi)}
            add(variant, b, float(mT[i].mean()) + (mb if correct else 0.0), K2 * (vg + vi), **extra)

    for b, i in blocks.items():
        add("A0", b, float(mT[i].mean()), K2 * float(np.sum(sT[i] ** 2)) / len(i) ** 2)
    parts = [("pop", POP, case["m_oof"]), ("flat", FLAT, case["m_oof"]), ("ins", POP, case["m_ins"])]
    for tag, prior, pred in parts:
        names = {"pop": ("J1pK", "A2", "A4", "A5", "A7", "A8"), "flat": ("A1", "A3"), "ins": ("A6",)}[tag]
        try:
            if pred is None:
                raise ValueError("missing cross-fitted or in-sample means")
            (avg, vbar), (plug, vbar_p), off = fit(prior, S, tr, te, y[tr] - pred, blocks)
        except Exception as e:                                                # counted and reported, never replaced
            for v in names:
                for b in blocks:
                    add(v, b, np.nan, np.nan, err=repr(e)[:150])
            continue
        if tag == "pop":
            assemble("J1pK", avg, vbar, gate, diag=off)
            assemble("A2", plug, vbar_p, gate)
            assemble("A4", avg, vbar, gate, off=off)
            assemble("A5", avg, vbar, gate, independent=False)
            assemble("A7", avg, vbar, False)
            assemble("A8", avg, vbar, True)
        elif tag == "flat":
            assemble("A1", avg, vbar, gate)
            assemble("A3", plug, vbar_p, gate)
        else:
            assemble("A6", avg, vbar, gate)
    return rows


def field_rows(job):
    set_name, field, out_dir, flt = job
    FILTER.update(flt); t0 = time.monotonic(); rows, missing = [], 0
    for f, data, design, rep, tr in block_cases.iter_splits(set_name, [field]):
        if (FILTER.get("designs") and design not in FILTER["designs"]) or (FILTER.get("reps") and rep not in FILTER["reps"]):
            continue
        for target in F.targets(f, set_name):
            if not np.all(np.isfinite(data["Y"][:, F.C.TARGETS.index(target)])):
                continue
            case = block_cases.load_case(set_name, f, data, design, rep, tr, target)
            if case is None:
                missing += 1; continue
            rows += case_rows(case)
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"ablation_{set_name}_{field}.csv", index=False)
    return field, len(rows), missing, time.monotonic() - t0


def check_against(r, stored_dir, set_name):
    """J1pK rows against the stored J1pK rows: blocks matched and the largest relative difference of M and V."""
    ref = pd.concat([pd.read_csv(f) for f in sorted(Path(stored_dir).glob(f"rows_{set_name}_*.csv"))], ignore_index=True)
    key = ["field", "design", "rep", "target", "block"]
    a = r[r.variant == "J1pK"].set_index(key); b = ref[ref.variant == "J1pK"].set_index(key)
    both = a.index.intersection(b.index)
    rel = max(float((np.abs(a.loc[both, c] - b.loc[both, c]) / np.abs(b.loc[both, c]).clip(lower=1e-12)).max()) for c in ("M", "V"))
    return len(both), len(a), len(b.index.difference(a.index)), rel


def report(set_name, out_dir, stored_dir):
    files = sorted(Path(out_dir).glob(f"ablation_{set_name}_*.csv")); r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    lines = []

    def say(s=""):
        lines.append(s); print(s, flush=True)

    say(f"=== Exploratory component ablation of TabPFN-GP, set '{set_name}' ===")
    matched, mine, ref_only, rel = check_against(r, stored_dir, set_name)
    say(f"check: J1pK vs the stored rows: {matched} of {mine} blocks matched, stored blocks not recomputed {ref_only}, "
        f"largest relative difference {rel:.1e}")
    keep = (r.src == "limesoda") | r[["field", "target"]].apply(tuple, axis=1).isin(F.PRIMARY_EXTERNAL)
    sc = scoring.score(r[keep]); lim = sc[sc.src == "limesoda"]; ext = sc[sc.src == "external"]
    say("\n-- failures (share of LimeSoDa blocks) --")
    for v in VARIANTS:
        s = lim[lim.variant == v]
        say(f"  {v:5s} {LABELS[v]:28s} failed {s.failed.mean():.4f} of {len(s)}")
    common = scoring.common_blocks(lim, list(VARIANTS)); cj = common[common.variant == "J1pK"]
    say(f"  common set: {int((cj.block == 'field').sum())} field blocks, {int((cj.block != 'field').sum())} zone blocks")
    say("\n-- LimeSoDa: coverage (95 % interval over fields) and CRPS ratio variant / J1pK (fields where the variant is better) --")
    for v in VARIANTS:
        parts = []
        for kind in ("field", "zone"):
            est, ci = scoring.pooled_coverage(common, v, kind)
            txt = f"{kind} cov {est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            if v != "J1pK":
                c = scoring.contrast(common, v, "J1pK", kind)
                txt += f", CRPS {c['ratio']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] ({c['better']}/{c['fields']})"
            parts.append(txt)
        say(f"  {v:5s} {LABELS[v]:28s} " + " | ".join(parts))
    ext_common = scoring.common_blocks(ext, list(VARIANTS))
    say("\n-- Oxford and Meuse, primary tasks: coverage and CRPS ratio variant / J1pK (interval from replicates; per task) --")
    for v in VARIANTS:
        parts = []
        for kind in ("field", "zone"):
            cov = float(scoring.kind_rows(ext_common, kind).query("variant == @v").cov80.mean())
            txt = f"{kind} cov {cov:.3f}"
            if v != "J1pK":
                e = scoring.external(ext_common, v, "J1pK", kind)
                txt += f", CRPS {e['ratio']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}] (" + ", ".join(f"{x:.3f}" for x in e["tasks"]) + ")"
            parts.append(txt)
        say(f"  {v:5s} {LABELS[v]:28s} " + " | ".join(parts))
    say("\n-- LimeSoDa, field means by n: coverage of each variant --")
    say("  n     " + " ".join(f"{v:>6s}" for v in VARIANTS))
    fc = scoring.kind_rows(common, "field")
    for n in sorted(fc.n.unique()):
        s = fc[fc.n == n]
        say(f"  {n:3d}   " + " ".join(f"{s[s.variant == v].cov80.mean():6.3f}" for v in VARIANTS))
    say("\n-- LimeSoDa, J1pK: median share of the block variance from the offset term and from the independent part, by n --")
    j = common[common.variant == "J1pK"]
    for n in sorted(j.n.unique()):
        s = j[j.n == n]; f_, z_ = s[s.block == "field"], s[s.block != "field"]
        say(f"  n = {n:3d}: field offset {f_.offset_share.median():.2f}, independent {f_.indep_share.median():.2f} | "
            f"zone offset {z_.offset_share.median():.2f}, independent {z_.indep_share.median():.2f}")
    Path(out_dir, f"ablation_report_{set_name}.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["rows", "report"])
    ap.add_argument("--set", required=True, choices=["first", "final"]); ap.add_argument("--out", required=True)
    ap.add_argument("--stored"); ap.add_argument("--fields", nargs="*"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--designs", nargs="*", help="tests only"); ap.add_argument("--reps", nargs="*", type=int, help="tests only")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.stage == "report":
        report(a.set, out, a.stored); return
    flt = {"designs": a.designs, "reps": a.reps}
    fields = a.fields or sorted({f for group, _, _ in F.SETS[a.set] for f in group})
    jobs = [(a.set, f, str(out), flt) for f in fields]
    with Pool(min(a.workers, len(jobs))) as pool:
        for field, n, missing, sec in pool.imap_unordered(field_rows, jobs):
            print(f"{a.set} {field}: {n} rows, {missing} cases without point outputs, {sec:.0f} s", flush=True)


if __name__ == "__main__":
    main()
