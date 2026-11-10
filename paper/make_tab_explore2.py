"""Tables C.4 and C.5 (exploratory analyses E3 and E4) from data/explore2.csv.

Run from figs/source:  python make_tab_explore2.py   -> ../../tab_bygroup_rise.tex, ../../tab_levels.tex
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = list(csv.DictReader(open(HERE / "data" / "explore2.csv")))
METHODS = ["TabPFN-GP", "Shared-error-only", "TabPFN", "TabPFN-KpR", "Block RK", "WH", "Sample mean"]
SHORT = {"TabPFN-GP": "GP", "Shared-error-only": "Shared", "TabPFN": "TabPFN", "TabPFN-KpR": "KpR", "Block RK": "RK", "WH": "WH",
         "Sample mean": "Mean"}
LABEL = {"Uniformity trials": "Uniformity trials", "SensRes": "SensRes", "Midwest": "Midwest",
         "New fields": "Canada, Czechia, Italy, USA, Switzerland", "Further fields": "France, USA, UK",
         "SOC_target": "SOC", "Clay_target": "Clay", "pH_target": "pH"}

# ---- rise D by group and property (tab_bygroup_rise)
e3 = {(r["by"], r["value"], r["method"], r["kind"]): r for r in R if r["analysis"] == "E3"}
out = [r"\begin{table}[!htbp]", r"\centering",
       r"\caption{\textbf{Rise in coverage by group of fields and by soil property} (exploratory). $D$: rise in coverage (Section~\ref{sec:evaluation}), over the fields"
       r" of the group or property with both $n \le 10$ and $n \ge 30$; $^{\circ}$: the $95\,\%$"
       r" bootstrap interval over these fields includes $0$. GP: TabPFN-GP; Shared: shared-error-only model; KpR: TabPFN-KpR; RK: block RK;"
       r" Mean: sample mean. The uniformity trials have two fields.}",
       r"\label{tab:rise_bygroup}", r"\scriptsize", r"\setlength{\tabcolsep}{1.8pt}",
       r"\begin{tabular}{@{}l" + "r" * len(METHODS) + "@{}}", r"\toprule",
       "& " + " & ".join(SHORT[m] for m in METHODS) + r" \\", r"\midrule"]
for kind in ("field", "zone"):
    out.append(r"\multicolumn{" + str(len(METHODS) + 1) + r"}{@{}l}{\emph{" + kind.capitalize() + r" means}} \\")
    for by, vals in (("group", ["Uniformity trials", "SensRes", "Midwest", "New fields", "Further fields"]),
                     ("property", ["SOC_target", "Clay_target", "pH_target"])):
        for v in vals:
            cells = []
            for m in METHODS:
                r = e3[(by, v, m, kind)]; d, lo, hi = float(r["D"]), float(r["lo"]), float(r["hi"])
                cells.append(f"${d:+.3f}$" + (r"$^{\circ}$" if lo <= 0 <= hi else ""))
            out.append(f"\\quad {LABEL[v]} & " + " & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(HERE.parent.parent / "tab_bygroup_rise.tex").write_text("\n".join(out) + "\n")

# ---- coverage at other levels, n <= 10 and n >= 30 (tab_levels)
e4 = {(r["method"], r["kind"], r["level"], r["band"]): r for r in R if r["analysis"] == "E4"}
out = [r"\begin{table}[!htbp]", r"\centering",
       r"\caption{\textbf{Coverage at other interval levels} (exploratory). Coverage of central $50$, $80$, $90$ and $95\,\%$ intervals,"
       r" each method with its single factor for all sample sizes; $\le 10$, $\ge 30$: with $n \le 10$ and with $n \ge 30$ revealed samples;"
       r" all: over all $n$. The column $\le 10$ pools all $43$ fields and $\ge 30$ the $40$ fields with enough locations, so their"
       r" difference is not the rise $D$, which uses the fields with both (Table~\ref{tab:scorecard}).}",
       r"\label{tab:levels}", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
       r"\begin{tabular}{@{}l" + "rrr" * 4 + "@{}}", r"\toprule",
       " & " + " & ".join(r"\multicolumn{3}{c}{$" + str(l) + r"\,\%$}" for l in (50, 80, 90, 95)) + r" \\",
       " ".join(r"\cmidrule(lr){" + f"{2 + 3 * k}-{4 + 3 * k}" + "}" for k in range(4)),
       "Method & " + " & ".join([r"$\le 10$ & $\ge 30$ & all"] * 4) + r" \\", r"\midrule"]
for kind in ("field", "zone"):
    out.append(r"\multicolumn{13}{@{}l}{\emph{" + kind.capitalize() + r" means}} \\")
    for m in METHODS:
        cells = ["$%.3f$" % float(e4[(m, kind, str(float(l)), b)]["coverage"]) for l in (50, 80, 90, 95) for b in ("n<=10", "n>=30", "all")]
        out.append(f"\\quad {m} & " + " & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(HERE.parent.parent / "tab_levels.tex").write_text("\n".join(out) + "\n")
print("written")
