"""The table of means over fewer and more locations of the same area (exploratory analysis E14), from data/dense_report.txt.

Run from figs/source:  python make_tab_dense.py   -> ../../tab_dense.tex
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = r"([+-]?[0-9.]+)"
CI = rf"{NUM} \[{NUM}, {NUM}\]"


def parse(lines):
    level, by_class, dense, kind = {}, {}, {}, None
    for ln in lines:
        if ln.startswith("-- field means"):
            kind = "field"
        elif ln.startswith("-- zone means"):
            kind = "zone"
        elif ln.startswith("-- dense limit"):
            kind = None
        m = re.match(rf"\s+(TabPFN-GP|Shared-error-only|TabPFN|Sample mean)\s+m =\s+10: {CI}\s+all: {CI}\s+D {CI}", ln)
        if m and kind:
            g = list(map(float, m.groups()[1:]))
            level[(kind, m.group(1))] = dict(c10=g[0], call=g[3], d=tuple(g[6:9]))
        m = re.match(rf"\s+m_B (\d+)-(\d+|max): D {CI}; coverage with all {NUM} \((\d+) areas, (\d+) fields\)", ln)
        if m and kind:
            by_class[(kind, int(m.group(1)))] = tuple(map(float, m.groups()[2:5])) + (int(m.group(8)),)   # D, its interval, fields
        m = re.match(rf"\s+(field|zone) m_B (\d+)-(\d+|max): independent share median {NUM} \(90th pct {NUM}\); coverage {CI}, "
                     rf"with V_inf {CI} \((\d+) areas, (\d+) fields\)", ln)
        if m:
            g = m.groups()
            dense[(g[0], int(g[1]))] = dict(hi=g[2], share=float(g[3]), cov=tuple(map(float, g[5:8])),
                                            vinf=tuple(map(float, g[8:11])), fields=int(g[12]))
    return level, by_class, dense


def ci(v, signed=False):
    e, lo, hi = v
    f = (lambda x: f"{x:+.3f}") if signed else (lambda x: f"{x:.3f}")
    return f"${f(e)}$ [${f(lo)}$, ${f(hi)}$]" if lo != hi else f"${f(e)}$"


def main():
    level, by_class, dense = parse((HERE / "data" / "dense_report.txt").read_text().splitlines())
    out = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{\textbf{Means over fewer and more locations of the same area} (exploratory). (a) Coverage of $80\,\%$ intervals of"
           r" the mean over $10$ randomly chosen unrevealed locations of an area and over all of them, on the same splits, with each"
           r" method's factor; difference: all minus $10$, over the areas with more than $10$ unrevealed locations. (b) TabPFN-GP by the number"
           r" of unrevealed locations $m_B$: difference as in (a); coverage with all locations; share of the variance from the independent part,"
           r" which averages out over more locations (median); coverage with that part removed, a lower bound for a mean over a dense grid"
           r" of the same area. Intervals: $95\,\%$ bootstrap intervals over fields; none where all areas are on one field.}",
           r"\label{tab:dense}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\vspace{0.6ex}",
           r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lllllll@{}}",
           r"\multicolumn{7}{@{}l}{\textbf{(a) Ten and all locations}} \\", r"\toprule",
           r" & \multicolumn{3}{c}{Field means} & \multicolumn{3}{c}{Zone means} \\",
           r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
           r"Method & $10$ & All & Difference & $10$ & All & Difference \\", r"\midrule"]
    for name in ("TabPFN-GP", "Shared-error-only", "TabPFN", "Sample mean"):
        cells = []
        for kind in ("field", "zone"):
            v = level[(kind, name)]
            cells += [f"${v['c10']:.3f}$", f"${v['call']:.3f}$", ci(v["d"], True)]
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular*}", "", r"\vspace{1.5ex}",
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llllll@{}}",
            r"\multicolumn{6}{@{}l}{\textbf{(b) TabPFN-GP by number of locations}} \\", r"\toprule",
            r"Means, $m_B$ & Fields & Difference & Coverage & \begin{tabular}[b]{@{}l@{}}Independent\\share\end{tabular} & "
            r"\begin{tabular}[b]{@{}l@{}}Coverage\\without it\end{tabular} \\", r"\midrule"]
    rows = (("field", 0, 0), ("field", 101, 101), ("field", 201, 201), ("zone", 3, None), ("zone", 10, 11), ("zone", 30, 30),
            ("zone", 100, 100))
    notes = []
    for kind, dk, ck in rows:
        d = dense[(kind, dk)]
        rng = (f"$\\le {d['hi']}$" if dk == 0 else f"$\\ge {dk}$" if d["hi"] == "max" else f"${dk}$--${d['hi']}$")
        dd = by_class.get((kind, ck)) if ck is not None else None
        dcell = ci(dd[:3], True) if dd else "--"
        if dd and dd[3] != d["fields"]:                          # D is over fewer fields than the coverage (no D where m_B = 10)
            dcell += r"$^{\ast}$"
            notes.append(rf" $^{{\ast}}$ (in b): difference over the ${dd[3]}$ fields with zones of ${ck}$ to ${d['hi']}$ locations (it is $0$ with $10$).")
        out.append(f"{kind.capitalize()}, {rng} & ${d['fields']}$ & " + dcell + f" & {ci(d['cov'])} & "
                   f"${d['share']:.3f}$ & {ci(d['vinf'])} \\\\")
    out += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    out[2] = out[2][:-1] + "".join(notes) + "}"                  # notes at the end of the caption
    (HERE.parent.parent / "tab_dense.tex").write_text("\n".join(out) + "\n")
    print("written")


if __name__ == "__main__":
    main()
