"""Figures of Appendix C: CRPS ratio by sample size (exploratory analyses E1 and E2, data/explore_by_n.csv), CRPS ratio by group of fields
and coverage by field (the study's report, data/report.txt).

Run from figs/source:  python fig_explore.py   -> ../fig_crps_by_n.pdf, ../fig_crps_by_group.pdf, ../fig_coverage_by_field.pdf
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from orx_figstyle import BASELINE, MUTED, PALETTE, figure_grid, panel_labels, save, use_style

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
NS = (5, 10, 15, 20, 30, 50, 75, 100)
COMP = {                                                      # comparator: colour, marker (as in fig_coverage_by_n)
    "TabPFN": (PALETTE["orange"], "s"),
    "TabPFN-KpR": (PALETTE["red"], "v"),
    "Block RK": (PALETTE["green"], "D"),
    "WH": (PALETTE["purple"], "^"),
    "Sample mean": (BASELINE, "x"),
}
REPORT = (HERE / "data" / "report.txt").read_text()


def crps_by_n():
    rows = defaultdict(dict)
    with open(HERE / "data" / "explore_by_n.csv") as f:
        for r in csv.DictReader(f):
            if int(r["n"]) > 0:
                rows[(r["analysis"], r["kind"], r["method"])][int(r["n"])] = tuple(float(r[c]) for c in ("crps_ratio", "crps_lo", "crps_hi"))
    fig, axes = figure_grid(2, 2, width=5.4, ratio=0.78, sharey=True, sharex=True)
    pos = {n: i for i, n in enumerate(NS)}
    for row, analysis in enumerate(("E1", "E2")):
        for col, kind in enumerate(("field", "zone")):
            ax = axes[row][col]
            ax.axhline(1.0, color=BASELINE, lw=0.6, zorder=0)
            for k, (m, (c, mk)) in enumerate(COMP.items()):
                d = rows[(analysis, kind, m)]
                x = [pos[n] + (k - 2) * 0.08 for n in NS if n in d]
                y = [d[n][0] for n in NS if n in d]
                err = [[d[n][0] - d[n][1] for n in NS if n in d], [d[n][2] - d[n][0] for n in NS if n in d]]
                ax.errorbar(x, y, yerr=err, color=c, marker=mk, ms=2.8, lw=1.0, elinewidth=0.6, capsize=0, label=m)
            ax.set_xticks(range(len(NS)), [str(n) for n in NS])
            ax.set_ylim(0.65, 1.08)
            if row == 1:
                ax.set_xlabel("revealed samples $n$")
        axes[row][0].set_ylabel("CRPS ratio, TabPFN-GP / method")
    panel_labels(axes, ["(a) field means, single factor for all $n$", "(b) zone means, single factor for all $n$",
                        "(c) field means, factor tuned for each $n$", "(d) zone means, factor tuned for each $n$"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, str(OUT / "fig_crps_by_n"))


def crps_by_group():
    groups = ("Uniformity trials", "SensRes", "Midwest", "New fields", "Further fields")
    labels = {"New fields": "Canada, Czechia, Italy,\nUSA, Switzerland", "Further fields": "France, USA, UK"}
    val, g = {}, None
    section = REPORT.split("by group of fields:")[1].split("interval-width")[0].split("\n")[1:]
    for ln in section:
        h = re.match(r"^  (\S.*) \(\d+ fields\)$", ln)
        if h:
            g = h.group(1); continue
        mm = re.match(r"^    (.+?)\s+coverage .*CRPS ratio ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", ln)
        if mm and mm.group(1) in COMP:
            val[(g, mm.group(1))] = [float(v) for v in mm.groups()[1:]]
    assert len(val) == len(groups) * len(COMP), len(val)
    fig, axes = figure_grid(1, 2, width=5.4, ratio=0.62, sharey=True)
    ypos = {}
    for gi, g in enumerate(groups):
        for k, m in enumerate(COMP):
            ypos[(g, m)] = -(gi * (len(COMP) + 1.5) + k)
    for ax, (kind, off) in zip(axes, (("field", 0), ("zone", 3))):
        ax.axvline(1.0, color=BASELINE, lw=0.6, zorder=0)
        for (g, m), y in ypos.items():
            e, lo, hi = val[(g, m)][off:off + 3]
            c, mk = COMP[m]
            ax.errorbar([e], [y], xerr=[[e - lo], [hi - e]], color=c, marker=mk, ms=3.0, lw=0, elinewidth=0.8, capsize=0,
                        label=m if g == groups[0] else None)
        ax.set_xlabel("CRPS ratio, TabPFN-GP / method")
        ax.set_xlim(0.6, 1.15)
    ticks = [np.mean([ypos[(g, m)] for m in COMP]) for g in groups]
    axes[0].set_yticks(ticks, [labels.get(g, g) for g in groups])
    axes[0].tick_params(axis="y", length=0)
    panel_labels(axes, ["(a) field means", "(b) zone means"])
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="outside lower center", ncol=5, frameon=False)
    save(fig, str(OUT / "fig_crps_by_group"))


def coverage_by_field():
    block = REPORT.split("coverage by field (field | zone):")[1].split("coverage by property")[0]
    methods = ["TabPFN-GP"] + list(COMP)
    cov = {m: {"field": [], "zone": []} for m in methods}
    for ln in block.strip().split("\n"):
        for m in methods:
            a, b = re.search(rf"(?:^|; |\s){re.escape(m)} ([0-9.]+) \| ([0-9.]+)", ln.strip().split(" ", 1)[1]).groups()
            cov[m]["field"].append(float(a)); cov[m]["zone"].append(float(b))
    fig, axes = figure_grid(1, 2, width=5.4, ratio=0.5, sharey=True)
    rng = np.random.default_rng(0)
    colours = {"TabPFN-GP": PALETTE["blue"], **{m: c for m, (c, _) in COMP.items()}}
    for ax, kind in zip(axes, ("field", "zone")):
        ax.axhspan(0.75, 0.85, color=MUTED, alpha=0.5, lw=0, zorder=0)
        ax.axhline(0.80, color=BASELINE, lw=0.6, zorder=0)
        for i, m in enumerate(methods):
            v = np.array(cov[m][kind])
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=colours[m], alpha=0.8, lw=0, zorder=2)
            inside = int(((v >= 0.75) & (v <= 0.85)).sum())
            ax.annotate(f"{inside}/43", (i, 0.30), ha="center", va="bottom", fontsize=6.5, color="#333333")
        ax.set_xticks(range(len(methods)), ["TabPFN-\nGP", "TabPFN", "TabPFN-\nKpR", "Block\nRK", "WH", "Sample\nmean"])
        ax.set_ylim(0.28, 1.0)
    axes[0].set_ylabel("coverage of 80 % intervals")
    panel_labels(axes, ["(a) field means", "(b) zone means"])
    save(fig, str(OUT / "fig_coverage_by_field"))


if __name__ == "__main__":
    use_style()
    crps_by_n()
    crps_by_group()
    coverage_by_field()
