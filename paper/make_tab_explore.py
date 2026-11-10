"""The table of factors for each sample size: tuned on the study's results (exploratory analysis E2, data/explore_by_n.csv) and
set on the factor-setting data (E13, data/report5.txt).

Run from figs/source:  python make_tab_explore.py   -> ../../tab_explore.tex
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = (5, 10, 15, 20, 30, 50, 75, 100)
METHODS = ("TabPFN", "TabPFN-KpR", "Block RK", "WH", "Sample mean")
rows = {}
with open(HERE / "data" / "explore_by_n.csv") as f:
    for r in csv.DictReader(f):
        if r["analysis"] == "E2" and int(r["n"]) > 0:
            rows[(r["kind"], r["method"], int(r["n"]))] = r
ext = {}                                                                      # E13: factor for each n, set on the factor-setting data
e13 = (HERE / "data" / "report5.txt").read_text().split("-- E13:")[1]
for m in METHODS:
    line = re.search(rf"^  {re.escape(m)}\s+factors: (.+)$", e13, re.M).group(1)
    ext[m] = {int(a): float(b) for a, b in re.findall(r"n=(\d+): ([0-9.]+)", line)}
    assert set(ext[m]) == set(NS), m

out = [r"\begin{table}[!htbp]", r"\centering",
       r"\caption{\textbf{Factors tuned for each sample size} (exploratory). Factor on the predictive standard deviation that minimised"
       r" each comparator's CRPS at each sample size $n$, and the smallest factor that gave a coverage of $0.80$ over all fields, on the grid"
       r" $0.05, 0.10, \dots, 20$. Single factors for all sample sizes (Table~\ref{tab:methods}): TabPFN $2.40$, TabPFN-KpR $4.20$, block RK $1.55$, WH $1.60$, sample mean"
       r" $1.05$. $^{\dagger}$TabPFN-KpR did not reach $0.80$ with a factor of $20$ (coverage $0.748$ and $0.739$ of field means with $5$ and $10$ samples,"
       r" $0.751$ of zone means with $10$). Last block: the factors of Table~\ref{tab:further}, set for each $n$ by the rule of"
       r" Section~\ref{sec:factors} on the factor-setting data, each task counted once (grid $0.50, 0.55, \dots, 5.00$).}",
       r"\label{tab:explore}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
       r"\begin{tabular}{@{}ll" + "r" * len(NS) + "@{}}", r"\toprule",
       r" & & \multicolumn{" + str(len(NS)) + r"}{c}{$n$} \\ \cmidrule(l){3-" + str(len(NS) + 2) + "}",
       "Factor & Method & " + " & ".join(f"${n}$" for n in NS) + r" \\", r"\midrule"]
for label, col in (("Lowest CRPS", "factor"), ("Coverage $0.80$", "cov_factor")):
    for kind in ("field", "zone"):
        out.append(r"\multicolumn{" + str(len(NS) + 2) + r"}{@{}l}{\emph{" + label + ", " + kind + r" means}} \\")
        for m in METHODS:
            cells = []
            for n in NS:
                r = rows[(kind, m, n)]
                v = float(r[col]); mark = r"$^{\dagger}$" if col == "cov_factor" and float(r["cov_at_factor"]) < 0.80 else ""
                cells.append(f"${v:.2f}${mark}")
            out.append(f" & {m} & " + " & ".join(cells) + r" \\")
out.append(r"\multicolumn{" + str(len(NS) + 2) + r"}{@{}l}{\emph{Coverage $0.80$ on the factor-setting data, field and zone means"
           r" (Table~\ref{tab:further})}} \\")
for m in METHODS:
    out.append(f" & {m} & " + " & ".join(f"${ext[m][n]:.2f}$" for n in NS) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(HERE.parent.parent / "tab_explore.tex").write_text("\n".join(out) + "\n")
print("written", HERE.parent.parent / "tab_explore.tex")
