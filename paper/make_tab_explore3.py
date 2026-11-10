"""The table of TabPFN-GP with normal and with Student-t intervals (exploratory analysis E6), from data/explore3.csv.

Run from figs/source:  python make_tab_explore3.py   -> ../../tab_tshape.tex
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = list(csv.DictReader(open(HERE / "data" / "explore3.csv")))
NAME = {"TabPFN-GP": "Normal", "TabPFN-GP, t shape": "Student-$t$"}


def get(what, method, kind, level="", band=""):
    m = [r for r in R if r["what"] == what and r["method"] == method and r["kind"] == kind
         and (level == "" or float(r["level"] or "nan") == level) and r["band"] == band]
    assert len(m) == 1, (what, method, kind, level, band, len(m))
    return m[0]


out = [r"\begin{table}[!htbp]", r"\centering",
       r"\caption{\textbf{Normal and Student-$t$ intervals of TabPFN-GP} (exploratory). Coverage of central intervals over all sample sizes"
       r" and, at $95\,\%$, with $n \le 10$ and $n \ge 30$; $D_{95}$: rise in the coverage of $95\,\%$ intervals from $n \le 10$ to"
       r" $n \ge 30$ [$95\,\%$ bootstrap interval]; CRPS: ratio Student-$t$ / normal. Student-$t$: $n - 1$ degrees of freedom, with the $80\,\%$ interval of the normal shape.}",
       r"\label{tab:tshape}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
       r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
       r"& \multicolumn{4}{c}{All $n$} & \multicolumn{2}{c}{$95\,\%$} & & \\",
       r"\cmidrule(lr){2-5} \cmidrule(lr){6-7}",
       r"& $50\,\%$ & $80\,\%$ & $90\,\%$ & $95\,\%$ & $n \le 10$ & $n \ge 30$ & $D_{95}$ & CRPS \\", r"\midrule"]
for kind in ("field", "zone"):
    out.append(r"\multicolumn{9}{@{}l}{\emph{" + kind.capitalize() + r" means}} \\")
    for m in ("TabPFN-GP", "TabPFN-GP, t shape"):
        cov = [f"${float(get('coverage', m, kind, l, 'all')['est']):.3f}$" for l in (50, 80, 90, 95)]
        cov += [f"${float(get('coverage', m, kind, 95, b)['est']):.3f}$" for b in ("n<=10", "n>=30")]
        d = get("rise", m, kind, 95)
        cov.append(f"${float(d['est']):+.3f}$ [${float(d['lo']):+.3f}$, ${float(d['hi']):+.3f}$]")
        cov.append("--" if m == "TabPFN-GP" else f"${float(get('crps_ratio', 'TabPFN-GP', kind)['est']):.3f}$")
        out.append(f"\\quad {NAME[m]} & " + " & ".join(cov) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(HERE.parent.parent / "tab_tshape.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
