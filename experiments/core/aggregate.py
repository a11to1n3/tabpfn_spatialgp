"""Aggregation of block-mean scores, shared by the scoring scripts (scoring.py).

Row = split x property x block. For a task (field x property), the value is log(mean score of a / mean score of b) over the task's rows,
the ratio of mean scores, a proper aggregation; a field's value is the mean over its properties; the pooled ratio is exp(mean over fields),
with a 95 % percentile bootstrap interval over fields (10,000, seed 0). The external datasets (one field each) are pooled over the four
primary tasks, with an interval from resampling replicates within each task (10,000, seed 0).
"""
import numpy as np
import pandas as pd

from experiments.core import datasets as C

B = 10000
BLOCK_KEY = ["field", "design", "rep", "target", "block"]


def pair(rows, a, b, col="crps"):
    x = rows[rows.variant == a].set_index(BLOCK_KEY)[col]; y = rows[rows.variant == b].set_index(BLOCK_KEY)[col].reindex(x.index)
    return pd.DataFrame({"a": x.values, "b": y.values}, index=x.index).reset_index().dropna()


def proper_by_field(t):
    task = t.groupby(["field", "target"])[["a", "b"]].apply(lambda q: np.log(q.a.mean() / q.b.mean()))
    return task.groupby(level="field").mean()


def boot_fields(Lf, seed=0):
    rng = np.random.default_rng(seed); v = Lf.values
    bs = np.exp(v[rng.integers(0, len(v), (B, len(v)))].mean(1))
    return float(np.exp(v.mean())), tuple(np.percentile(bs, [2.5, 97.5]))


def external_contrast(rows, a, b, kind="field", seed=0):
    """a vs b on the primary external tasks: per task log(sum a / sum b) over all rows; CI by resampling replicates within each task."""
    s = rows[(rows.block == "field") == (kind == "field")]; t = pair(s, a, b); rng = np.random.default_rng(seed); logs, boots = [], []
    for field, target in C.EXTERNAL_TASKS:
        q = t[(t.field == field) & (t.target == target)]
        if q.empty:
            continue
        reps = np.sort(q.rep.unique()); A = q.groupby("rep").a.sum().reindex(reps).values; Bv = q.groupby("rep").b.sum().reindex(reps).values
        logs.append(np.log(A.sum() / Bv.sum())); idx = rng.integers(0, len(reps), (B, len(reps))); boots.append(np.log(A[idx].sum(1) / Bv[idx].sum(1)))
    return {"ratio": float(np.exp(np.mean(logs))), "ci": tuple(np.percentile(np.exp(np.mean(boots, 0)), [2.5, 97.5])),
            "tasks": [float(np.exp(x)) for x in logs]}
