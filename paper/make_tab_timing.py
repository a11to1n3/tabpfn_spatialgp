"""The table of computing time of every method per split (exploratory analysis E5), from data/timing_methods.csv.

Run from figs/source:  python make_tab_timing.py   -> ../../tab_timing.tex
"""
import csv
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
R = list(csv.DictReader(open(HERE / "data" / "timing_methods.csv")))
METHODS = ["TabPFN", "TabPFN-GP", "WH", "Block RK", "TabPFN-KpR", "Sample mean"]
WORK = {"TabPFN": "$1$ & --", "TabPFN-GP": "$6$ & --", "WH": "$6$ & correlogram", "Block RK": "$1$ & block kriging",
        "TabPFN-KpR": "$1$ & leave-one-out kriging", "Sample mean": "$0$ & --"}


def q(x, p):
    x = sorted(x); k = (len(x) - 1) * p; i = int(k)
    return x[i] + (x[min(i + 1, len(x) - 1)] - x[i]) * (k - i)


OK = {"WH": "WH_ok", "Block RK": "RK_ok", "TabPFN-KpR": "KpR_ok"}


def num(x):
    return "$<0.001$" if x < 0.001 else f"${x:.2f}$"


def rows_of(m):
    """Splits on which the method succeeded (a failed call is not a time for the method)."""
    return [r for r in R if m not in OK or r[OK[m]] == "True"]


out = [r"\begin{table}[!htbp]", r"\centering",
       r"\caption{\textbf{Computing time per split} (exploratory; Section~\ref{sec:disc-limitations}). Median and $10$--$90\,\%$ range"
       r" over $" + str(len(R)) + r"$ splits; ratio: median over splits of the method's time divided by TabPFN's. TabPFN-KpR's"
       r" leave-one-out kriging failed on $" + str(len(R) - len(rows_of("TabPFN-KpR"))) + r"$ splits with $n = 5$, left out. WH was timed with the five fold fits, which its correlogram, taken from the development fields, does not need on a new field.}",
       r"\label{tab:timing}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{@{}lrlrrr@{}}", r"\toprule",
       r"Method & TabPFN fits & In R & Median (s) & $10$--$90\,\%$ (s) & Ratio \\", r"\midrule"]
for m in METHODS:
    rr = rows_of(m); x = [float(r[m]) for r in rr]; ratio = median(float(r[m]) / float(r["TabPFN"]) for r in rr)
    out.append(f"{m} & {WORK[m]} & {num(median(x))} & "
               + (num(q(x, 0.9)) if q(x, 0.9) < 0.001 else f"{num(q(x, 0.1))}--{num(q(x, 0.9))}") + " & "
               + (f"${ratio:.1f}$" if ratio >= 0.01 else "$<0.01$") + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(HERE.parent.parent / "tab_timing.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
for n in sorted({int(r["n"]) for r in R}):
    g = [r for r in R if int(r["n"]) == n]
    print(n, len(g), "; ".join(f"{m} {median(float(r[m]) for r in g if r in rows_of(m)):.3f}" for m in METHODS))
print("failures:", {k: sum(r[k] != "True" for r in R) for k in ("WH_ok", "RK_ok", "KpR_ok")})
