"""TabPFN-GP's block model (J1p): the predictive distribution of field and zone means from the saved point outputs.
Used by block_cases.case_rows and the exploratory analyses. CPU only.

block_model fits the residual GP with a population prior to TabPFN's cross-fitted residuals, takes its joint posterior over each block,
and returns {block: (M, V, GP mean correction)}: M = TabPFN's block mean (+ the GP's mean correction if gate, from n = 25 in the study), V = the GP's
block variance + sum of max(0, v_T - v_bar) / m^2. J1pK in the factor-setting rows is J1p with V multiplied by kappa^2 = 1.10^2 (config.KAPPA_OURS).
"""
import json

import numpy as np

from experiments.paths import CONFIGS
from tabpfn_spatialgp.gp_field import ResidualField, block_range_floor, field_extent

PRIORS = json.loads((CONFIGS / "priors.json").read_text())["block"]   # 'POP' (the population prior) and 'LOFO'


def block_model(prior, S, tr, te, y, ours, blocks, sT, gate):
    """Our block model with the given population prior: {block: (M, V, GP mean correction)}."""
    g = ResidualField(prior, field_extent(S)).fit(S[tr], y[tr] - ours["bb_m_oof"], block_range_floor(S[tr]))
    moments, vbar = g.block_moments(S[te], blocks)
    independent = np.clip(sT ** 2 - vbar, 0, None); mT = ours["none_mean"]; out = {}
    for b, i in blocks.items():
        mb, vb = moments[b]
        out[b] = ((float(mT[i].mean()) + mb) if gate else float(mT[i].mean()), float(np.sum(independent[i])) / len(i) ** 2 + vb, mb)
    return out
