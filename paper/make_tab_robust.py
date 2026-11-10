"""Tables C.8 and C.9 (exploratory robustness checks and further methods) from the study's reports.

Run from figs/source:  python make_tab_robust.py   -> ../../tab_robust.tex, ../../tab_further.tex
Inputs in data/: report.txt (the study), report_k110.txt (kappa = 1.10), report_once.txt (factors by one rule), report5.csv
(datasets, further methods, factors for each n), report7.csv (TabPFN-GP without its factor), report8.csv (comparators' variogram settings from fewer fields), and report_seed1..4.txt
(seeds 1-4) when present.
"""
import csv
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COMP = ("TabPFN", "TabPFN-KpR", "Block RK", "WH", "Sample mean")
NUM = r"([+-]?[0-9.]+)"


def parse(path):
    """{quantity: (estimate, lo, hi)} from the endpoints of a study report."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(rf"H1 (\S+(?: \S+)?)\s+field-mean D = {NUM}, 95 % \[{NUM}, {NUM}\]", line)
        if m:
            out[f"t1 {m.group(1).strip()}"] = tuple(map(float, m.groups()[1:]))
        m = re.match(rf"H2 TabPFN-GP (field|zone)-mean D = {NUM}, 90 % \[{NUM}, {NUM}\]", line)
        if m:
            out[f"t2 {m.group(1)}"] = tuple(map(float, m.groups()[1:]))
        m = re.match(rf"H3 TabPFN-GP (field|zone)-mean coverage {NUM} \[{NUM}, {NUM}\]", line)
        if m:
            out[f"t3 {m.group(1)}"] = tuple(map(float, m.groups()[1:]))
        m = re.match(rf"H4\(a\) shared-error-only field-mean coverage {NUM} \[{NUM}, {NUM}\]; CRPS ratio to TabPFN-GP {NUM} \[{NUM}, {NUM}\]", line)
        if m:
            g = list(map(float, m.groups())); out["t4a cov"] = tuple(g[:3]); out["t4a crps"] = tuple(g[3:])
        m = re.match(rf"H4\(b\) TabPFN-GP minus shared-error-only zone-mean coverage {NUM}, 95 % \[{NUM}, {NUM}\]", line)
        if m:
            out["t4b"] = tuple(map(float, m.groups()))
        m = re.match(rf"H5 (field|zone)\s+TabPFN-GP / (.+?)\s+CRPS ratio {NUM} \[{NUM}, {NUM}\].*: (better|no worse|not met)", line)
        if m:
            out[f"t5 {m.group(1)} {m.group(2).strip()}"] = tuple(map(float, m.groups()[2:5])) + (m.group(6),)
    return out


ROWS = ([(f"t1 {c}", f"1: $D$, {c}") for c in COMP[:4]] +
        [("t2 field", "2: $D$, field means"), ("t2 zone", "2: $D$, zone means"),
         ("t3 field", "3: coverage, field means"), ("t3 zone", "3: coverage, zone means"),
         ("t4a cov", "4a: coverage"), ("t4a crps", "4a: CRPS ratio"),
         ("t4b", "4b: coverage difference")] +
        [(f"t5 field {c}", f"5, field: {c}") for c in COMP] + [(f"t5 zone {c}", f"5, zone: {c}") for c in COMP])


def cell(v, signed=False, mark=""):
    """Estimate over its interval, two lines in one cell."""
    if v is None:
        return "--"
    e, lo, hi = v[:3]
    f = (lambda x: f"{x:+.3f}") if signed else (lambda x: f"{x:.3f}")
    return (r"\begin{tabular}[c]{@{}c@{}}" + f"${f(e)}${mark}" + r"\\[-0.4ex]{\tiny $[" + f"{f(lo)}, {f(hi)}" + r"]$}\end{tabular}")


def main():
    base, k110, once = parse(DATA / "report.txt"), parse(DATA / "report_k110.txt"), parse(DATA / "report_once.txt")
    r5 = list(csv.DictReader(open(DATA / "report5.csv")))
    ds = {}
    for r in r5:
        if r["analysis"] != "E9.1":
            continue
        w, k = r["what"], r["kind"]
        key = {"t1": f"t1 {r['model']}", "t2": f"t2 {k}", "t3": f"t3 {k}", "t4a_cov": "t4a cov", "t4a_crps": "t4a crps", "t4b": "t4b",
               "t5": f"t5 {k} {r['model']}"}[w]
        holm = float(r["n"]) if w == "t5" and r["n"] else None
        ds[key] = (float(r["est"]), float(r["lo"]), float(r["hi"]), holm)
    seeds = [parse(DATA / f"report_seed{s}.txt") for s in (1, 2, 3, 4) if (DATA / f"report_seed{s}.txt").exists()]
    two = lambda a, b: r"\begin{tabular}[b]{@{}c@{}}" + a + r"\\" + b + r"\end{tabular}"
    cols = ["As tested", two("$17$ source", "datasets"), "$\\kappa = 1.10$", two("One factor", "rule")] + ([two("Five seeds", "(mean; range)")] if seeds else [])
    out = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{\textbf{Robustness of the five tests} (exploratory). Estimates with $95\,\%$ intervals ($90\,\%$ for test 2) from"
           r" bootstrap resamples of the $43$ fields (as tested, $\kappa = 1.10$, one factor rule) or of the $17$ source datasets."
           r" $\kappa = 1.10$: TabPFN-GP's factor by the rule with the comparators' weighting of the factor-setting data. One factor rule:"
           r" every method's factor, the shared-error-only model's included ($1.20$), by the rule with each task counted once;"
           r" CRPS of the other methods with or without their factor, as the lower-CRPS version rule of Section~\ref{sec:factors} chose (without for the shared-error-only model),"
           r" and TabPFN-GP's with $\kappa = 1.15$."
           + (r" Five seeds: seeds $0$ to $4$ of TabPFN and of the fold split, all methods rerun; mean (geometric mean for ratios) and range."
              if seeds else "") +
           r" Rows: test (Section~\ref{sec:evaluation}) and quantity; 4a: the shared-error-only model, its CRPS ratio to TabPFN-GP. $^{\circ}$: the criterion of the test is not met; $^{\ast}$: no worse instead of"
           r" better (test 5).}",
           r"\label{tab:robust}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.0}",
           r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule", "Test and quantity & " + " & ".join(cols) + r" \\", r"\midrule"]

    def met(key, v, sig):
        if v is None:
            return ""
        e, lo, hi = v[:3]
        if key.startswith("t1"):
            return "" if lo > 0 else "$^{\\circ}$"
        if key.startswith("t2"):
            return "" if -0.10 < lo and hi < 0.10 else "$^{\\circ}$"
        if key.startswith("t3") or key == "t4a cov":
            return "" if lo >= 0.75 and hi <= 0.85 else "$^{\\circ}$"
        if key == "t4a crps":
            return "" if hi < 1.02 else "$^{\\circ}$"
        if key == "t4b":
            return "" if lo > 0 else "$^{\\circ}$"
        if key.startswith("t5"):
            verdict = sig
            return "" if verdict == "better" else ("$^{\\ast}$" if verdict == "no worse" else "$^{\\circ}$")
        return ""

    for key, label in ROWS:
        signed = key.startswith(("t1", "t2", "t4b"))
        cells = []
        for src in (base, ds, k110, once):
            v = src.get(key)
            sig = None
            if key.startswith("t5") and v is not None:
                sig = v[3] if isinstance(v[3], str) else ("better" if (v[3] is not None and v[3] < 0.05 and v[0] < 1) else
                                                         ("no worse" if v[2] < 1.02 else "not met"))
            cells.append(cell(v, signed, met(key, v, sig)))
        if seeds:
            vals = [base.get(key)] + [s.get(key) for s in seeds]
            es = [v[0] for v in vals if v is not None]
            ratio = key.startswith(("t5", "t4a crps"))
            m = math.exp(sum(math.log(x) for x in es) / len(es)) if ratio else sum(es) / len(es)
            f = (lambda x: f"${x:+.3f}$") if signed else (lambda x: f"${x:.3f}$")
            marks = [met(key, v, (v[3] if key.startswith("t5") else None)) for v in vals if v is not None]
            nmet = sum(1 for mk in marks if mk == "")
            cells.append(r"\begin{tabular}[c]{@{}c@{}}" + f"{f(m)}" + ("" if nmet == len(es) else f" ({nmet}/{len(es)} met)")
                         + r"\\[-0.4ex]{\tiny " + f"{f(min(es))} to {f(max(es))}" + r"}\end{tabular}")
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE.parent.parent / "tab_robust.tex").write_text("\n".join(out) + "\n")

    # ---- further methods (tab_further)
    get = {}
    for r in r5:
        if r["analysis"] in ("E11", "E13"):
            get[(r["analysis"], r["variant"], r["what"], r["kind"])] = (float(r["est"]), float(r["lo"]), float(r["hi"]))
        if r["analysis"] == "E10":
            get[("E10", r["variant"], "factor", "")] = (float(r["est"]),)
    for r in csv.DictReader(open(DATA / "report7.csv")):
        if r["analysis"] == "E15" and r["what"] in ("coverage", "D", "crps_ratio"):
            get[("E15", r["variant"], r["what"], r["kind"])] = (float(r["est"]), float(r["lo"]), float(r["hi"]))
    for r in csv.DictReader(open(DATA / "report8.csv")):
        if r["set"] in ("X2", "X3") and r["comparator"] in ("Block RK", "WH"):
            if r["what"] == "factor":
                get[("E18", r["set"] + r["comparator"], "factor", "")] = (float(r["est"]),)
            elif r["what"] in ("coverage", "D", "crps_ratio"):
                get[("E18", r["set"] + r["comparator"], r["what"], r["kind"])] = (float(r["est"]), float(r["lo"]), float(r["hi"]))
    groups = [("Other methods, each with its own factor", "E11",
               [("DE", "Difference estimator (field means)"), ("DEz", "Difference estimator (zone means)"),
                ("RK_oof_fix_v", "Block RK of held-out errors"), ("WH_loc", "WH with each field's own correlogram"),
                ("Z", "Zone-effect model")]),
              ("TabPFN-GP without some components, each with its own factor", "E11",
               [("GPx", "Prior refitted without SL.125, UL.120"), ("GPnp", "No cross-field prior"), ("GPna", "No averaging"),
                ("GPnn", "Neither"), ("J1p_1", r"Without its factor$^{\ddagger}$", "E15")]),
              ("Comparators with a factor for each $n$, set on the factor-setting data", "E13",
               [("TAB", "TabPFN"), ("KpR4", "TabPFN-KpR"), ("RK_ins_fix_v", "Block RK"), ("WH_fix", "WH"), ("DB", "Sample mean")]),
              ("Block RK and WH with variogram settings from fewer development fields; | factors and CRPS versions by the rule of the main comparison", "E18",
               [("X2Block RK", "Block RK, without SL.125, UL.120"), ("X2WH", "WH, without SL.125, UL.120"),
                ("X3Block RK", "Block RK, also without SP.231"), ("X3WH", "WH, also without SP.231")])]
    out = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{\textbf{Further methods and components} (exploratory). Factors by the rule of Section~\ref{sec:factors} with each"
           r" task counted once; coverage of $80\,\%$ intervals; $D$: rise in coverage; CRPS ratio TabPFN-GP / method, the"
           r" CRPS of both by the version rule (with or without the factor, whichever had the lower CRPS on the $20$ factor-setting fields; for"
           r" comparators with a factor for each $n$ (Table~\ref{tab:explore}), each keeps its version of Table~\ref{tab:methods} at every $n$ and TabPFN-GP's CRPS is as in Table~\ref{tab:tests}). $^{\dagger}$: $95\,\%$ interval"
           r" excludes $0$; $^{\circ}$: $95\,\%$ interval includes $1$; $^{\ddagger}$: no factor ($\kappa = 1$), CRPS ratio TabPFN-GP"
           r" with $\kappa = 1.15$, as in Table~\ref{tab:tests}, over TabPFN-GP with $\kappa = 1$.}",
           r"\label{tab:further}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
           r" & & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{$D$} & \multicolumn{2}{c}{CRPS ratio} \\",
           r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
           r"Method & Factor & field & zone & field & zone & field & zone \\", r"\midrule"]
    for title, an, items in groups:
        out += [r"\multicolumn{8}{@{}l}{\emph{" + part + r"}} \\" for part in title.split(" | ")]        # a long heading takes two lines
        for item in items:
            v, label, a = item if len(item) == 3 else (*item, an)
            fac = (1.0,) if a == "E15" else get.get(("E18" if a == "E18" else "E10", v, "factor", ""))
            ftxt = "per $n$" if a == "E13" else (f"${fac[0]:.2f}$" if fac else "--")
            row = [ftxt]
            for what in ("coverage", "D", "crps_ratio"):
                for k in ("field", "zone"):
                    x = get.get((a, v, what, k))
                    if x is None:
                        row.append("--"); continue
                    if what == "D":
                        row.append(f"${x[0]:+.3f}$" + ("$^{\\dagger}$" if x[1] > 0 or x[2] < 0 else ""))
                    elif what == "crps_ratio":
                        row.append(f"${x[0]:.3f}$" + ("$^{\\circ}$" if x[1] <= 1 <= x[2] else ""))
                    else:
                        row.append(f"${x[0]:.3f}$")
            out.append(f"\\quad {label} & " + " & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE.parent.parent / "tab_further.tex").write_text("\n".join(out) + "\n")
    print("written")


if __name__ == "__main__":
    main()
