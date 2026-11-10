"""E20 of docs/exploratory/plan_E20.md (CPU): what is subtracted from TabPFN's variance, on the development splits.

  SOILGP_DATA=.. SOILGP_RUN=.. python -m experiments.exploratory.variance_subtraction rows <out> [workers]
      rows of the four versions and check 2 of the plan
  SOILGP_DATA=.. SOILGP_RUN=.. python -m experiments.exploratory.variance_subtraction report <out> <stored rows>
      check 1 of the plan and the report
<stored rows>: the folder with the rows of the other alternatives of the table of settings (nugget_dev_<field>.csv, written by development/dev_nugget.py).
"""
import copy
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
from multiprocessing import Pool                                              # noqa: E402
from pathlib import Path                                                      # noqa: E402

MODE, OUT = sys.argv[1], Path(sys.argv[2])
ARG = sys.argv[3] if len(sys.argv) > 3 else None

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from scipy.linalg import cho_factor, cho_solve                                # noqa: E402
from scipy.spatial.distance import cdist                                      # noqa: E402
from experiments.core import config as F                                      # noqa: E402
from experiments.core import block_cases                                      # noqa: E402
from experiments.core import scoring                                          # noqa: E402
from experiments.development.ablation import POP                              # noqa: E402
from experiments.factor_setting.set_factors import factor_for                 # noqa: E402
from tabpfn_spatialgp import settings as S_                                   # noqa: E402
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent, matern32   # noqa: E402

VERSIONS = ["base", "marginal", "loo_unknown", "local"]
ALTS = VERSIONS[1:]
KINDS = ("field", "zone")
KEYS = ["field", "design", "rep", "target", "block", "variant"]
DEV = list(F.C.DEV_FIELDS)
MARGIN = 0.005                                                                # the 0.5 % of the comparators' choice of version
RISE_LIMIT = 0.05


def dev_cases(field):                                                         # as in development/development.py
    for f, data, design, rep, tr in block_cases.iter_splits("dev", [field], min_test=1):
        for target in F.targets(f, "dev"):
            if not np.all(np.isfinite(data["Y"][:, F.C.TARGETS.index(target)])):
                continue
            case = block_cases.load_case("dev", f, data, design, rep, tr, target)
            if case is not None:
                yield case


def per_theta(g, th, Dq):
    """The candidate shares at one grid point, in the units of the standardised residuals."""
    p = g._unpack(th); _, Fa, _, _ = g._nlp(th, ret=True); n = len(g.y)
    Ki = cho_solve(Fa, np.eye(n)); one = np.ones(n); Ki1 = Ki @ one; A = float(one @ Ki1)
    P = Ki - np.outer(Ki1, Ki1) / A
    kq = p["s1"] * matern32(Dq, p["l1"]); Kikq = cho_solve(Fa, kq.T); u = 1.0 - (g.H.T @ Kikq).ravel()
    return dict(p=p, Fa=Fa, Ki=Ki, P=P, A=A, kq=kq, Kikq=Kikq, u=u, diag=p["s1"] - np.sum(kq * Kikq.T, 1) + u ** 2 / A,
                vbar=float(np.mean(np.clip(1 / np.diag(Ki) - p["tau"], 0, None))),
                vP=float(np.mean(np.clip(1 / np.diag(P) - p["tau"], 0, None))))


def shares(g, Sq):
    """The amounts subtracted in the four versions, in the original units (`local` is one value per target location)."""
    t = [per_theta(g, th, cdist(Sq, g.S)) for th in g.thetas]; s2 = g.ys ** 2; w = g.w
    return dict(base=s2 * float(np.dot(w, [x["vbar"] for x in t])), marginal=s2 * float(np.dot(w, [x["p"]["s1"] for x in t])),
                loo_unknown=s2 * float(np.dot(w, [x["vP"] for x in t])), local=s2 * (w[:, None] * np.array([x["diag"] for x in t])).sum(0))


def close(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(np.abs(a), np.abs(b))))


def check2(g, Sq, blocks, field):
    """Check 2 of the plan, for every grid point of one split."""
    n = len(g.y); Dqq = cdist(Sq, Sq); one = np.ones(n - 1)
    for th in g.thetas:
        x = per_theta(g, th, cdist(Sq, g.S)); p = x["p"]; s2 = g.ys ** 2
        assert close(x["vbar"], g._vbar_theta(x["Fa"], p), 1e-12), ("check 2: v_bar", field)
        K = p["s1"] * matern32(g.D, p["l1"]) + (p["tau"] + S_.JITTER) * np.eye(n)
        direct = np.empty(n)
        for i in range(n):
            idx = np.delete(np.arange(n), i); Fr = cho_factor(K[np.ix_(idx, idx)], lower=True); k = K[idx, i]
            a = cho_solve(Fr, k); b = cho_solve(Fr, one)
            direct[i] = K[i, i] - k @ a + (1.0 - one @ a) ** 2 / (one @ b)
        assert close(1 / np.diag(x["P"]), direct, 1e-6), ("check 2: leave-one-out variance", field)
        assert np.all(1 / np.diag(x["P"]) >= (1 / np.diag(x["Ki"])) * (1 - 1e-12)), ("check 2: ordering", field)
        one_point = copy.copy(g); one_point.thetas, one_point.w = [th], np.array([1.0])
        assert close(s2 * x["diag"], one_point.predict_points(Sq)[1], 1e-8), ("check 2: point variance", field)
        C = p["s1"] * matern32(Dqq, p["l1"]) - x["kq"] @ x["Kikq"] + np.outer(x["u"], x["u"]) / x["A"]
        moments = one_point.block_moments(Sq, blocks)[0]
        for b_name, i in blocks.items():
            assert close(s2 * C[np.ix_(i, i)].mean(), moments[b_name][1], 1e-8), ("check 2: block variance", field, b_name)


def rows_field(field):
    t0 = time.monotonic(); rows, amounts, checked = [], [], False
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
            sh = shares(g, S[te])
        except Exception as e:
            for v in VERSIONS:
                for b in blocks:
                    add(v, b, np.nan, np.nan, err=repr(e)[:150])
            continue
        assert close(sh["base"], vbar, 1e-9), ("check 2: mixed v_bar", field, sh["base"], vbar)
        if not checked and 10 <= n <= 40:
            check2(g, S[te], blocks, field); checked = True
        amounts.append(dict(field=field, design=case["design"], rep=case["rep"], target=case["target"], n=n, vbar=sh["base"], marginal=sh["marginal"],
                            loo_unknown=sh["loo_unknown"], local=float(np.mean(sh["local"]))))
        for v in VERSIONS:
            ind = np.clip(sT ** 2 - sh[v], 0, None)
            for b, i in blocks.items():
                mb, vgb = moments[b]; m = len(i)
                add(v, b, float(mT[i].mean()) + (mb if gate else 0.0), vgb + float(np.sum(ind[i])) / m ** 2)
    assert checked, ("check 2: no split with 10 to 40 samples", field)
    pd.DataFrame(rows).to_csv(OUT / f"vbar_dev_{field}.csv", index=False)
    pd.DataFrame(amounts).to_csv(OUT / f"amounts_dev_{field}.csv", index=False)
    return field, len(rows), time.monotonic() - t0


def evaluate(sub, factors):
    r = scoring.score(sub, factors); common = scoring.common_blocks(r, VERSIONS)
    cov = {v: [float(scoring.kind_rows(common[common.variant == v], kd).cov80.mean()) for kd in KINDS] for v in VERSIONS}
    return cov, {alt: [scoring.contrast(common, alt, "base", kd) for kd in KINDS] for alt in ALTS}


def fmt(c):
    return "-" if c is None else f"{c['ratio']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] ({c['better']}/{c['fields']} fields lower)"


def report(stored):
    lines, table = [], []

    def say(s=""):
        lines.append(s); print(s, flush=True)

    rows = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(OUT.glob("vbar_dev_*.csv"))], ignore_index=True)
    rows["failed"] = rows.failed.astype(bool)
    new = rows[rows.variant == "base"]
    old = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(Path(stored).glob("nugget_dev_*.csv"))], ignore_index=True)
    old = old[old.variant == "base"]
    assert rows.field.nunique() == 14 and old.field.nunique() == 14, "check 1: fields"
    m = old.merge(new, on=KEYS, suffixes=("_o", "_n"), validate="one_to_one"); assert len(m) == len(old) == len(new) == 6954, ("check 1: blocks", len(m), len(old), len(new))
    assert not new.failed.any(), "check 1: failures"
    assert (m.M_o - m.M_n).abs().max() <= 1e-9 and (m.V_o - m.V_n).abs().max() <= 1e-9, ("check 1: values", (m.M_o - m.M_n).abs().max(), (m.V_o - m.V_n).abs().max())
    expected = {1.0: (0.780, 0.765), 1.10: (0.827, 0.807)}
    for k, e in expected.items():
        cov, _ = evaluate(rows, {v: k for v in VERSIONS})
        assert (round(cov["base"][0], 3), round(cov["base"][1], 3)) == e, ("check 1: coverage", k, cov["base"], e)
    print("check 1 passed: base reproduces the stored rows, 6954 blocks, coverage as stored", flush=True)

    own = {v: factor_for(rows, v)[0] for v in VERSIONS}
    settings = [("kappa = 1.00", {v: 1.0 for v in VERSIONS}), ("kappa = 1.10", {v: 1.10 for v in VERSIONS}), ("own factor", own)]
    z = rows.block != "field"
    bands = [("all", rows)] + [(f"n {lo}-{hi if hi < 10 ** 6 else 'max'}", rows[rows.n.between(lo, hi)]) for lo, hi in ((5, 10), (15, 20), (30, 10 ** 6))]
    bands += [(f"zones m_B {lo}-{hi if hi < 10 ** 6 else 'max'} (field column = all field means)", rows[(~z) | rows.mB.between(lo, hi)]) for lo, hi in ((3, 9), (10, 29), (30, 10 ** 6))]

    say("=== E20 (docs/exploratory/plan_E20.md; specified after the study's results had been seen; exploratory): what is subtracted from TabPFN's variance ===")
    say("14 development fields, 6954 blocks per version; base = TabPFN-GP as it is; CRPS ratio = version / base")
    say("own factors: " + ", ".join(f"{v} {own[v]:.2f}" for v in VERSIONS))
    results = {}
    for name, fac in settings:
        say(f"\n-- {name} (field | zone) --")
        for lab, sub in bands:
            cov, con = evaluate(sub, fac); results[name, lab] = (cov, con)
            say(f"  {lab}: coverage base {cov['base'][0]:.3f} | {cov['base'][1]:.3f}")
            for alt in ALTS:
                say(f"    {alt:12s} coverage {cov[alt][0]:.3f} | {cov[alt][1]:.3f}; CRPS {fmt(con[alt][0])} | {fmt(con[alt][1])}")
                for j, kd in enumerate(KINDS):
                    c = con[alt][j]
                    table.append(dict(setting=name, band=lab, version=alt, kind=kd, factor_base=fac["base"], factor_version=fac[alt], coverage_base=cov["base"][j], coverage_version=cov[alt][j],
                                      ratio=np.nan if c is None else c["ratio"], lo=np.nan if c is None else c["ci"][0], hi=np.nan if c is None else c["ci"][1],
                                      lower=np.nan if c is None else c["better"], fields=np.nan if c is None else c["fields"]))

    amounts = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("amounts_dev_*.csv"))], ignore_index=True)
    good = amounts[amounts.vbar > 0]
    say(f"\n-- amount subtracted, as a ratio to v_bar: median [quartiles] over {len(good)} cases (v_bar = 0 in {int((amounts.vbar <= 0).sum())}) --")
    for lab, lo, hi in (("all", 0, 10 ** 6), ("n 5-10", 5, 10), ("n 15-20", 15, 20), ("n 30-max", 30, 10 ** 6)):
        s = good[good.n.between(lo, hi)]
        say(f"  {lab}: " + "; ".join(f"{alt} {np.median(s[alt] / s.vbar):.2f} [{np.quantile(s[alt] / s.vbar, .25):.2f}, {np.quantile(s[alt] / s.vbar, .75):.2f}]" for alt in ALTS))

    say("\n-- reading (fixed before running, docs/exploratory/plan_E20.md), own factor --")
    _, con = results["own factor", "all"]
    for alt in ALTS:
        for j, kd in enumerate(KINDS):
            c = con[alt][j]
            verdict = "clearly better" if c["ratio"] < 1 - MARGIN and c["ci"][1] < 1 else "clearly worse" if c["ratio"] > 1 + MARGIN and c["ci"][0] > 1 else "no clear difference"
            say(f"  {alt:12s} {kd:5s}: {verdict} ({fmt(c)})")
    low, high = results["own factor", "n 5-10"][0], results["own factor", "n 30-max"][0]
    rise = {v: high[v][0] - low[v][0] for v in VERSIONS}
    say("  coverage of field means, n 30 or more minus n 5 to 10: " + ", ".join(f"{v} {rise[v]:+.3f}" for v in VERSIONS))
    for alt in ALTS:
        say(f"  {alt:12s} differs from base by {rise[alt] - rise['base']:+.3f}: " + ("more than 0.05, stated in the row" if abs(rise[alt] - rise["base"]) > RISE_LIMIT else "within 0.05"))
    (OUT / "report10.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(table).to_csv(OUT / "report10.csv", index=False)
    (OUT / "DONE").touch()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if MODE == "rows":
        with Pool(int(ARG or 6)) as pool:
            for field, k, s in pool.imap_unordered(rows_field, DEV):
                print(f"rows {field}: {k} rows, {s:.0f} s", flush=True)
        print("check 2 passed on every field", flush=True)
    elif MODE == "report":
        report(ARG)
    else:
        sys.exit(__doc__)
