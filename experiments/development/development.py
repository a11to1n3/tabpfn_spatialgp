"""TabPFN-GP's development choices on the 14 development fields (plan: docs/development/plan_development.md).

  python -m experiments.development.development blocks --out DIR [--workers 14]
      choice 5: block rows of the base model and its alternatives on the development splits (kappa = 1; CPU, from the stored point
      outputs)
  python -m experiments.development.development screen --out DIR [--workers 14]
      choice 4: point SSE of the screen's GP variants refitted with the point prior, and of the stored point GP (for the cutoff); CPU
  python -m experiments.development.development report --out DIR --abl DIR12 DIR2 --ver DIR12 DIR2 --v35 DIR12 DIR2
      every choice: choices 1-3 from the TabPFN outputs of the development comparisons on the 12 LimeSoDa fields (DIR12, read-only)
      and on LYD.112 and RUM.70 (DIR2) -> DIR/report_development.txt
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                               # noqa: E402
import re                                                                     # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from scipy.stats import wilcoxon                                              # noqa: E402

from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.development.ablation import POP, fit                         # noqa: E402
from experiments.core.point_model import point_prior                          # noqa: E402
from tabpfn_spatialgp.gp_field import field_extent, point_range_floor        # noqa: E402

DEV = list(F.C.DEV_FIELDS)
SMALL, LARGE = ["random5", "random10", "random20", "random30"], ["random50", "random70", "random90"]
SCREEN_DESIGNS = ["random10", "random30", "random50", "random70"]
CUT_DESIGNS = ["random10", "random30", "random50", "random70", "random90"]
BINS = [(0, 25), (25, 50), (50, 100), (100, 10 ** 6)]
BASE_GATE_N = 50  # choice 5's base model: the mean correction from n = 50 (plan_development.md); the study uses F.GATE_N
SCREEN = {"G0": {}, "Ghet": dict(het=True), "Gdrift": dict(drift=True), "Gtwo": dict(two=True), "Gall": dict(het=True, drift=True, two=True),
          "Gbma": dict(bma=True), "Ghetbma": dict(het=True, bma=True), "Gbma0": dict(bma=True, null=True), "Gbmashr": dict(bma=True, shrink=True),
          "Gbma0shr": dict(bma=True, null=True, shrink=True), "G0shr": dict(shrink=True)}
BLOCK_VARIANTS = {"base": "base model", "no_offset": "without the offset term", "no_split": "without the split of TabPFN's variance",
                  "no_avg": "without averaging", "no_corr": "without the mean correction", "corr_all": "mean correction at every n"}
B = 10000
RNG = np.random.default_rng(20261017)


def dev_cases(field):
    for f, data, design, rep, tr in block_cases.iter_splits("dev", [field], min_test=1):
        for target in F.targets(f, "dev"):
            if not np.all(np.isfinite(data["Y"][:, F.C.TARGETS.index(target)])):
                continue
            case = block_cases.load_case("dev", f, data, design, rep, tr, target)
            if case is not None:
                yield case


# ------------------------------------------------------------------------------------------------------------------------- choice 5
def block_rows(field_out):
    field, out = field_out; t0 = time.monotonic(); rows = []
    for case in dev_cases(field):
        S, tr, te, y = case["S"], case["tr"], case["te"], case["y"]; n = len(tr); yq = y[te]
        blocks, revealed = block_cases.blocks_of(case); mT, sT = case["mT"], case["sT"]; gate = n >= BASE_GATE_N

        def add(v, b, M, V, err=""):
            i = blocks[b]; failed = bool(err) or not (np.isfinite(M) and np.isfinite(V) and V > 1e-18)
            rows.append(dict(set="dev", src="limesoda", field=field, design=case["design"], rep=case["rep"], target=case["target"], n=n,
                             m=len(te), block=b, mB=len(i), nB=len(revealed[b]), variant=v, M=float(M), V=float(V), df=np.inf,
                             obs=float(yq[i].mean()), failed=failed, err=err))
        try:
            (avg, vbar), (plug, vbar_p), off = fit(POP, S, tr, te, y[tr] - case["m_oof"], blocks)
        except Exception as e:                                                # counted and reported
            for v in list(BLOCK_VARIANTS) + ["corr_25"]:
                for b in blocks:
                    add(v, b, np.nan, np.nan, err=repr(e)[:150])
            continue
        for v, moments, vb, correct, offset, split in (("base", avg, vbar, gate, False, True), ("no_offset", avg, vbar, gate, True, True),
                                                       ("no_split", avg, vbar, gate, False, False), ("no_avg", plug, vbar_p, gate, False, True),
                                                       ("no_corr", avg, vbar, False, False, True), ("corr_all", avg, vbar, True, False, True),
                                                       ("corr_25", avg, vbar, n >= 25, False, True)):        # exploratory (not in the plan)
            ind = np.clip(sT ** 2 - vb, 0, None) if split else sT ** 2
            for b, i in blocks.items():
                mb, vgb = moments[b]; m = len(i)
                add(v, b, float(mT[i].mean()) + (mb if correct else 0.0), vgb - (off[b] if offset else 0.0) + float(np.sum(ind[i])) / m ** 2)
    pd.DataFrame(rows).to_csv(Path(out) / f"blocks_dev_{field}.csv", index=False)
    return field, len(rows), time.monotonic() - t0


# ------------------------------------------------------------------------------------------------------------------------- choice 4
def screen_rows(field_out):
    from experiments.development.gp_variants import SpatialField
    field, out = field_out; t0 = time.monotonic(); rows = []
    for case in dev_cases(field):
        design = case["design"]; n = len(case["tr"]); o = case["ours"]
        if design not in set(SCREEN_DESIGNS) | set(CUT_DESIGNS):
            continue
        S, tr, te, y = case["S"], case["tr"], case["te"], case["y"]; yq = y[te]; mT = case["mT"]
        row = dict(field=field, design=design, rep=case["rep"], target=case["target"], n=n, sse_none=float(np.sum((mT - yq) ** 2)),
                   sse_stored_gp=float(np.sum((o["Gbma_mean"] - yq) ** 2)))
        if design in SCREEN_DESIGNS:
            prior = point_prior(field, case["target"]); ext = field_extent(S); floor = point_range_floor(S[tr], prior, ext)
            for vn, kw in SCREEN.items():
                if n < 8:
                    row[f"sse_{vn}"] = row["sse_none"]; continue
                kw = dict(kw); bma = kw.pop("bma", False)
                try:
                    g = SpatialField(prior=prior, ext=ext, **kw).fit(S[tr], y[tr] - case["m_oof"], v=case["v_oof"], min_len=floor, bma=bma)
                    row[f"sse_{vn}"] = float(np.sum((mT + g.predict(S[te])[0] - yq) ** 2))
                except Exception:                                             # counted in the report, TabPFN alone in its place
                    row[f"sse_{vn}"] = row["sse_none"]; row[f"fail_{vn}"] = True
        rows.append(row)
    pd.DataFrame(rows).to_csv(Path(out) / f"screen_dev_{field}.csv", index=False)
    return field, len(rows), time.monotonic() - t0


# ------------------------------------------------------------------------------------------------------------------------- statistics
def summarize(L):
    """Field values (log ratios) -> pooled ratio, 95 % field-cluster bootstrap interval, two-sided Wilcoxon, fields better (ablation)."""
    x = np.asarray(L, float); x = x[np.isfinite(x)]; boot = np.exp(np.array([x[RNG.integers(len(x), size=len(x))].mean() for _ in range(B)]))
    try:
        p = float(wilcoxon(x).pvalue)
    except ValueError:
        p = 1.0
    return {"ratio": float(np.exp(x.mean())), "ci": (float(np.quantile(boot, .025)), float(np.quantile(boot, .975))), "p": p,
            "better": int((x < 0).sum()), "fields": len(x)}


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def tabpfn_cells(roots, pattern_models, designs, reps):
    """SSE per cell from TabPFN outputs (<root>/<design>/<field>/<target>/<model>_rep<r>.npz, variant 'none'), new runs first."""
    rows = []
    for root in roots:
        for p in Path(root).glob("*/*/*/*_rep*.npz"):
            design, field, target = p.parts[-4], p.parts[-3], p.parts[-2]; m, rep = re.fullmatch(r"(.+)_rep(\d+)\.npz", p.name).groups()
            if m not in pattern_models or design not in designs or int(rep) not in reps or field not in DEV:
                continue
            with np.load(p, allow_pickle=True) as z:
                rows.append(dict(field=field, target=target, design=design, rep=int(rep), model=m, root=str(root),
                                 sse=float(np.sum((z["none_mean"] - z["observed"]) ** 2))))
    df = pd.DataFrame(rows).drop_duplicates(["field", "target", "design", "rep", "model"], keep="first")
    return df.pivot_table(index=["field", "target", "design", "rep"], columns="model", values="sse", aggfunc="first")


def field_values(W, M, R, designs):
    M, R = [M] if isinstance(M, str) else M, [R] if isinstance(R, str) else R
    sub = W[W.index.get_level_values("design").isin(designs)]
    d = (np.log(sub[M]).mean(1) - np.log(sub[R]).mean(1)).dropna()
    return d.groupby(level="field").mean().reindex(DEV)


def fmt(s):
    return f"{s['ratio']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}], {s['better']}/{s['fields']} fields, p = {s['p']:.3g}"


# ------------------------------------------------------------------------------------------------------------------------- report
def report(a):
    out = Path(a.out); lines = []
    def say(s=""):
        lines.append(s); print(s, flush=True)
    say("=== TabPFN-GP's development choices on the 14 development fields (plan_development.md) ===")
    say("fields: " + ", ".join(DEV))
    verdicts = {}

    # choices 1 and 2: ablation blocks A and C, small band
    W = tabpfn_cells([a.abl[1], a.abl[0]], ["O-xy4", "O-xy-def", "O-xy-n4", "O-xy-noT"] + [f"FT-{r}-s{k}" for r in ("syn", "real", "mix") for k in range(3)],
                     SMALL + LARGE, range(15, 20))
    say(f"\n-- choices 1 and 2: TabPFN's pipeline and weights (cells {len(W)}; models complete: "
        + ", ".join(f"{m} {int(W[m].notna().sum())}" for m in W.columns) + ") --")
    S3 = lambda r: [f"FT-{r}-s{k}" for k in range(3)]
    for block, cons in (("A pipeline", [("A1 four members instead of eight", "O-xy-n4", "O-xy-def"), ("A2 no target transform", "O-xy-noT", "O-xy-def"),
                                        ("A3 both (xy4 against TabPFN's default)", "O-xy4", "O-xy-def")]),
                        ("C fine-tuning against the released weights (seed-averaged)", [("C1 synthetic GP data", S3("syn"), "O-xy4"),
                                                                                       ("C2 real data", S3("real"), "O-xy4"), ("C3 real and synthetic data", S3("mix"), "O-xy4")])):
        for band, designs in (("small", SMALL), ("large", LARGE)):
            res = [(name, summarize(field_values(W, M, R, designs)), M, R) for name, M, R in cons]
            adj = holm(np.array([r[1]["p"] for r in res]))
            say(f"  {block} | {band} band")
            for (name, s, M, R), h in zip(res, adj):
                s["holm"] = float(h); extra = ""
                if isinstance(M, list):
                    pairs = [float(np.exp(field_values(W, m, R, designs).mean())) for m in M]
                    s["seed_robust"] = all(v < 1 for v in pairs) or all(v > 1 for v in pairs); extra = f"; seeds {min(pairs):.3f} to {max(pairs):.3f}"
                say(f"    {name}: SSE {fmt(s)}, Holm {h:.3g}{extra}")
                if band == "small":
                    verdicts[name[:2]] = s
    a3, a1 = verdicts["A3"], verdicts["A1"]
    keep_pipe = a3["ratio"] < 1 and a3["holm"] < 0.05; keep_four = not (a1["ratio"] > 1 and a1["holm"] < 0.05)
    ft = [verdicts[k] for k in ("C2", "C3")]
    adopt_ft = any(s["ratio"] < 1 and s["holm"] < 0.05 and s.get("seed_robust", False) for s in ft)
    say(f"  -> choice 1, pipeline: {'confirmed' if keep_pipe else 'CHANGES'} (no target transform); four members: {'confirmed' if keep_four else 'CHANGES'}")
    say(f"  -> choice 2, released weights: {'CHANGES (fine-tuning adopted)' if adopt_ft else 'confirmed'}")

    # choice 3: version 3.5 against 2.0.9 (backbones), random designs
    V = tabpfn_cells([a.ver[1], a.ver[0]], ["SIMPLE"], SMALL + LARGE, range(25, 30)).join(
        tabpfn_cells([a.v35[1], a.v35[0]], ["S35"], SMALL + LARGE, range(25, 30)), how="inner")
    s = summarize(field_values(V, "S35", "SIMPLE", SMALL + LARGE)); verdicts["version"] = s
    say(f"\n-- choice 3: TabPFN 3.5 against 2.0.9, both with four members and no target transform, without the GP (cells {len(V)}) --")
    say(f"  random 5-90 %: SSE {fmt(s)}")
    for band, designs in (("small", SMALL), ("large", LARGE)):
        say(f"  {band} band: SSE {fmt(summarize(field_values(V, 'S35', 'SIMPLE', designs)))}")
    say(f"  -> choice 3, version 2.0.9: {'CHANGES' if s['ratio'] < 1 and s['ci'][1] < 1 else 'confirmed'}")

    # choice 4: the screen
    sc = pd.concat([pd.read_csv(p) for p in sorted(out.glob("screen_dev_*.csv"))], ignore_index=True)
    say(f"\n-- choice 4: GP variants of the screen on the chosen backbone, point prior (cells "
        f"{int(sc.design.isin(SCREEN_DESIGNS).sum())}; fit failures "
        + ", ".join(f"{v} {int(sc.get(f'fail_{v}', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())}" for v in SCREEN) + ") --")
    say("  mean over tasks of log(SSE variant / SSE TabPFN), SSE summed over replicates; shown as ratios")
    wins, beaten = 0, False
    for d in SCREEN_DESIGNS:
        q = sc[sc.design == d].groupby(["field", "target"])[[f"sse_{v}" for v in SCREEN] + ["sse_none"]].sum()
        lr = {v: float(np.mean(np.log(q[f"sse_{v}"] / q["sse_none"]))) for v in SCREEN}; best = min(lr, key=lr.get)
        wins += best == "Gbma"; beaten |= any(lr[v] < lr["Gbma"] + np.log(0.99) for v in SCREEN)
        say(f"  {d}: " + "; ".join(f"{v} {np.exp(lr[v]):.3f}" for v in SCREEN) + f" -> best {best}")
    keep_avg = wins >= 3 or not beaten
    say(f"  -> choice 4, averaging: {'confirmed' if keep_avg else 'CHANGES'} (best at {wins} of 4 designs; beaten by more than 1 % at any: {beaten})")

    # choice 5: block model, kappa = 1 (and 1.10)
    rows = pd.concat([pd.read_csv(p) for p in sorted(out.glob("blocks_dev_*.csv"))], ignore_index=True); rows["failed"] = rows.failed.astype(bool)
    say(f"\n-- choice 5: block model on the development splits (blocks per variant {int((rows.variant == 'base').sum())}; failures "
        f"{int(rows[rows.variant == 'base'].failed.sum())}) --")
    keep_block = True
    for k in (1.0, F.KAPPA_OURS):
        r = scoring.score(rows, {v: k for v in BLOCK_VARIANTS}); common = scoring.common_blocks(r, list(BLOCK_VARIANTS))
        say(f"  kappa = {k:.2f}: coverage (field | zone) and CRPS ratio variant / base (field | zone)")
        cov = {v: [float(scoring.kind_rows(common[common.variant == v], kd).cov80.mean()) for kd in ("field", "zone")] for v in BLOCK_VARIANTS}
        say(f"    base model: coverage {cov['base'][0]:.3f} | {cov['base'][1]:.3f}")
        for v in list(BLOCK_VARIANTS)[1:]:
            c = [scoring.contrast(common, v, "base", kd) for kd in ("field", "zone")]
            better = all(ci["ratio"] < 1 for ci in c) and all(abs(cov[v][j] - 0.8) <= abs(cov["base"][j] - 0.8) for j in range(2))
            if k == 1.0 and better:
                keep_block = False
            say(f"    {BLOCK_VARIANTS[v]}: coverage {cov[v][0]:.3f} | {cov[v][1]:.3f}; CRPS {c[0]['ratio']:.3f} [{c[0]['ci'][0]:.3f}, {c[0]['ci'][1]:.3f}]"
                f" | {c[1]['ratio']:.3f} [{c[1]['ci'][0]:.3f}, {c[1]['ci'][1]:.3f}]" + ("  (better on the rule)" if better else ""))
    say(f"  -> choice 5, block model: {'confirmed' if keep_block else 'CHANGES'}")
    say("  exploratory (not in the plan): the mean correction from n = 25 against the base model (from n = 50), kappa = 1")
    r = scoring.score(rows, {v: 1.0 for v in list(BLOCK_VARIANTS) + ["corr_25"]}); common = scoring.common_blocks(r, ["base", "corr_25"])
    for lab, q in (("all n", common), ("25 <= n < 50", common[(common.n >= 25) & (common.n < 50)])):
        c = [scoring.contrast(q, "corr_25", "base", kd) for kd in ("field", "zone")]
        cv = [[float(scoring.kind_rows(q[q.variant == v], kd).cov80.mean()) for kd in ("field", "zone")] for v in ("base", "corr_25")]
        say(f"    {lab}: CRPS {c[0]['ratio']:.3f} [{c[0]['ci'][0]:.3f}, {c[0]['ci'][1]:.3f}] | {c[1]['ratio']:.3f} [{c[1]['ci'][0]:.3f}, {c[1]['ci'][1]:.3f}];"
            f" coverage {cv[0][0]:.3f} -> {cv[1][0]:.3f} | {cv[0][1]:.3f} -> {cv[1][1]:.3f}")

    # choice 5: cutoff, point level
    c = sc[sc.design.isin(CUT_DESIGNS)]
    say("\n-- choice 5: cutoff of the mean correction, point GP against TabPFN, SSE by n bin (10, 30, 50, 70 and 90 % designs) --")
    ratios = {}
    for lo, hi in BINS:
        q = c[(c.n >= lo) & (c.n < hi)]
        if q.empty:
            continue
        L = q.groupby("field").apply(lambda d: np.log(d.sse_stored_gp.sum() / d.sse_none.sum())); s = summarize(L.to_numpy())
        ratios[lo] = s["ratio"]
        say(f"  n in [{lo}, {hi if hi < 10 ** 6 else 'inf'}): {len(q)} cells, {s['fields']} fields; SSE {fmt(s)}; pooled over cells "
            f"{q.sse_stored_gp.sum() / q.sse_none.sum():.3f}")
    edges = [e for e in (25, 50, 100) if all(ratios.get(lo, 1.0) < 0.98 for lo, _ in BINS if lo >= e and lo in ratios)]
    e = min(edges) if edges else None
    say(f"  -> choice 5, cutoff: smallest edge with every bin above it below 0.98: {e}; cutoff of 50 {'confirmed' if e == 50 else 'CHANGES'}")
    (out / "report_development.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["blocks", "screen", "report"]); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14); ap.add_argument("--fields", nargs="*")
    ap.add_argument("--abl", nargs=2); ap.add_argument("--ver", nargs=2); ap.add_argument("--v35", nargs=2)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage == "report":
        report(a); return
    fn = block_rows if a.stage == "blocks" else screen_rows
    with Pool(a.workers) as pool:
        for field, n, s in pool.imap_unordered(fn, [(f, a.out) for f in (a.fields or DEV)]):
            print(f"{a.stage} {field}: {n} rows, {s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
