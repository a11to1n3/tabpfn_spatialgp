"""The table of other sampling designs (exploratory), from data/report11.csv (expected/exploratory/report11.csv).

Run from figs/source:  python make_tab_designs.py   -> ../../tab_designs.tex
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = list(csv.DictReader(open(HERE / "data" / "report11.csv")))
DESIGNS = ["strat", "clust", "tcov", "tsoil"]          # against the study's random samples, n = 10 to 30, 43 fields
METHODS = ["TabPFN-GP", "Shared-error-only", "TabPFN", "Block RK", "WH", "Sample mean", "Difference estimator", "Stratified estimator"]
LABEL = {"Shared-error-only": "Shared-error-only model"}


def pooled(design, kind, method, col):
    """The pooled row (all sizes) of one design, kind and method that carries `col`, or None."""
    for r in R:
        if r["design"] == design and r["n"] == "" and r["kind"] == kind and r["method"] == method and r[col] != "":
            return r
    return None


def reference(kind, method, designs):
    """Coverage of the reference random samples: a design's coverage less its paired change, the same for every design given."""
    vals = []
    for d in designs:
        r = pooled(d, kind, method, "coverage")
        if r is not None and r["change"] not in ("", "nan"):
            vals.append(float(r["coverage"]) - float(r["change"]))
    if not vals:
        return None
    assert max(vals) - min(vals) < 1e-9, (kind, method, vals)
    return vals[0]


def cov(x):
    if x is None:
        return "--"
    return f"$\\mathbf{{{x:.3f}}}$" if 0.75 <= x <= 0.85 else f"${x:.3f}$"


def ratio(r):
    return "--" if r is None else f"${float(r['crps_ratio']):.3f}$"


def main():
    cols = ["Random", "Stratified", "Clustered", "Covariate", "Property", "Random", "Grid"]
    out = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{\textbf{Other sampling designs} (exploratory). (a, b) Coverage of $80\,\%$ intervals with each method's factor:"
           r" on the $43$ fields, pooled over $n = 10$, $20$ and $30$, random samples of the study and samples stratified by zone,"
           r" clustered, or targeted on a covariate or on the property; on the $15$ fields with a recovered lattice, grid samples and random"
           r" samples of the same size. Bold: within $[0.75, 0.85]$. (c, d) CRPS ratio TabPFN-GP / method on the same samples; below $1$"
           r" favours TabPFN-GP. Designs and $95\,\%$ intervals: Section~\ref{sec:explore}.}",
           r"\label{tab:designs}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
           r" & \multicolumn{5}{c}{$43$ fields, $n = 10$ to $30$} & \multicolumn{2}{c}{$15$ fields} \\",
           r"\cmidrule(lr){2-6} \cmidrule(lr){7-8}",
           r" & & & & \multicolumn{2}{c}{Targeted on} & & \\", r"\cmidrule(lr){5-6}",
           "Method & " + " & ".join(cols) + r" \\", r"\midrule"]
    for kind, title in (("field", "(a) Coverage, field means"), ("zone", "(b) Coverage, zone means")):
        out.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for m in METHODS:
            if kind == "zone" and m == "Stratified estimator":
                continue
            cells = [cov(reference(kind, m, DESIGNS))]
            for d in DESIGNS:
                r = pooled(d, kind, m, "coverage"); cells.append(cov(None if r is None else float(r["coverage"])))
            cells.append(cov(reference(kind, m, ["grid"])))
            r = pooled("grid", kind, m, "coverage"); cells.append(cov(None if r is None else float(r["coverage"])))
            out.append(rf"\quad {LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    for kind, title in (("field", "(c) CRPS ratio, field means"), ("zone", "(d) CRPS ratio, zone means")):
        out.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for m in METHODS[1:7]:
            cells = ["--"] + [ratio(pooled(d, kind, m, "crps_ratio")) for d in DESIGNS] + ["--", ratio(pooled("grid", kind, m, "crps_ratio"))]
            out.append(rf"\quad {LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE.parent.parent / "tab_designs.tex").write_text("\n".join(out) + "\n")
    print("written", HERE.parent.parent / "tab_designs.tex")


if __name__ == "__main__":
    main()
