"""Coverage of 80 % intervals across sample sizes, field and zone means (the study's report, coverage_by_n.csv).
Regenerate: python paper/fig_coverage_by_n.py [coverage_by_n.csv] [output stem]."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from orx_figstyle import BASELINE, MUTED, PALETTE, figure_grid, panel_labels, save, use_style

HERE = Path(__file__).resolve().parent
NS = (5, 10, 15, 20, 30, 50, 75, 100)
STYLE = {                                                     # method: colour, line style, marker
    "TabPFN-GP": (PALETTE["blue"], "-", "o"),
    "Shared-error-only": (PALETTE["cyan"], "--", "o"),
    "TabPFN": (PALETTE["orange"], "-", "s"),
    "TabPFN-KpR": (PALETTE["red"], "-", "v"),
    "Block RK": (PALETTE["green"], "-", "D"),
    "WH": (PALETTE["purple"], "-", "^"),
    "Sample mean": (BASELINE, ":", "x"),
}


def load(path):
    rows = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(r["kind"], r["method"])][int(r["n"])] = (float(r["coverage"]), float(r["lo"]), float(r["hi"]))
    return rows


def main():
    use_style()
    rows = load(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "expected" / "study" / "coverage_by_n.csv")
    fig, axes = figure_grid(1, 2, width=5.4, ratio=0.46, sharey=True)
    pos = {n: i for i, n in enumerate(NS)}
    for ax, kind in zip(axes, ("field", "zone")):
        ax.axhspan(0.75, 0.85, color=MUTED, alpha=0.5, lw=0, zorder=0)
        ax.axhline(0.80, color=BASELINE, lw=0.6, zorder=0)
        for k, (method, (col, ls, mk)) in enumerate(STYLE.items()):
            d = rows[(kind, method)]
            x = [pos[n] + (k - 3) * 0.06 for n in NS if n in d]
            y = [d[n][0] for n in NS if n in d]
            lo = [d[n][0] - d[n][1] for n in NS if n in d]
            hi = [d[n][2] - d[n][0] for n in NS if n in d]
            ax.errorbar(x, y, yerr=[lo, hi], color=col, ls=ls, marker=mk, ms=2.8, lw=1.0, elinewidth=0.6, capsize=0,
                        label=method, zorder=3 if method == "TabPFN-GP" else 2)
        ax.set_xticks(range(len(NS)), [str(n) for n in NS])
        ax.set_xlabel("revealed samples $n$")
        ax.set_ylim(0.25, 1.0)
    axes[0].set_ylabel("coverage of 80 % intervals")
    panel_labels(axes, ["(a) field means", "(b) zone means"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    save(fig, sys.argv[2] if len(sys.argv) > 2 else "fig_coverage_by_n")


if __name__ == "__main__":
    main()
