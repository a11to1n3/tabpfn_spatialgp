"""The table of zones formed from covariates and from the property (exploratory), from data/report12.csv and data/report13.csv
(expected/exploratory/report12.csv, report13.csv).

Run from figs/source:  python make_tab_covzones.py   -> ../../tab_covzones.tex
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANELS = [("(a) Zones formed from covariates", list(csv.DictReader(open(HERE / "data" / "report12.csv")))),
          ("(b) Zones formed from the property", list(csv.DictReader(open(HERE / "data" / "report13.csv"))))]
METHODS = ["TabPFN-GP", "Shared-error-only", "TabPFN", "TabPFN-KpR", "Block RK", "WH", "Sample mean", "Difference estimator",
           "Zone-effect model"]
LABEL = {"Shared-error-only": "Shared-error-only model"}


def row(R, section, method, subset):
    for r in R:
        if r["section"] == section and r["method"] == method and r["subset"] == subset:
            return r
    return None


def num(x, bold=False, sign=False):
    if x in (None, "", "nan"):
        return "--"
    v = float(x); v = 0.0 if abs(v) < 5e-4 else v                             # no negative zero
    s = f"{v:+.3f}" if sign and v != 0 else f"{v:.3f}"
    return f"$\\mathbf{{{s}}}$" if bold and 0.75 <= v <= 0.85 else f"${s}$"


def main():
    out = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{\textbf{Zones formed from covariates and from the property} (exploratory). Coverage of $80\,\%$ intervals of zone"
           r" means with each method's factor, on the study's splits of the $43$ fields: (a) four zones per field from $k$-means of its"
           r" covariates' principal components; (b) four zones per field and property from $k$-means of the property's mean at each"
           r" location's four nearest other locations; and the study's compact zones; change paired by field; by the number of revealed"
           r" samples in the zone; rise in coverage $D$ ($n \ge 30$ minus $n \le 10$); the sample mean and the difference estimator need"
           r" two samples in the zone; CRPS ratio TabPFN-GP / method, below $1$ favours TabPFN-GP. All columns but the compact zones' refer"
           r" to the zones of the panel. Bold: within $[0.75, 0.85]$. $95\,\%$ intervals: Section~\ref{sec:explore}.}",
           r"\label{tab:covzones}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
           r" & \multicolumn{3}{c}{Coverage} & \multicolumn{3}{c}{Revealed samples in zone} & & \\",
           r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
           r"Method & Zones & Compact & Change & None & $1$--$2$ & $\ge 3$ & $D$ & CRPS ratio \\"]
    for title, R in PANELS:
        out += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for m in METHODS:
            c = row(R, "coverage", m, "all")
            cells = [num(c["coverage"], True), num(c["compact"], True), num(c["change"], sign=True)]
            for sub in ("none", "one or two", "three or more"):
                r = row(R, "revealed", m, sub); cells.append(num(None if r is None else r["coverage"], True))
            r = row(R, "rise", m, "D"); cells.append(num(r["coverage"], sign=True))
            r = row(R, "crps_ratio", m, "all"); cells.append("--" if r is None else num(r["coverage"]))
            out.append(rf"{LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE.parent.parent / "tab_covzones.tex").write_text("\n".join(out) + "\n")
    print("written", HERE.parent.parent / "tab_covzones.tex")


if __name__ == "__main__":
    main()
