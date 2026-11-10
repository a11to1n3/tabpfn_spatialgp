"""Priors of the residual GP.

Both priors are Gaussian on two quantities:
    log(range / E)          ~ N(mu_r, s_r^2),   E = field extent = sqrt(area of the convex hull of the field's sample locations)
    logit(spatial share)    ~ N(mu_s, s_s^2),   spatial share = sigma^2 / (sigma^2 + tau^2)

POINT_PRIORS (point predictions, one per property): empirical-Bayes priors from pilot GP fits on four fields of the KpR paper
(BB.250, BB.72, G.150 and MG.112). Every other field receives this prior; each of the four received the prior from the other three
(experiments/development/derive_priors.py derives both).

BLOCK_PRIOR (field and zone means, all properties): population prior from 630 development GP fits (n >= 50, one per split and property)
on the 14 development fields (configs/prior_fits.csv; experiments/development/derive_priors.py). It lets the GP be fitted from 3 training points
(evaluated from 5). A leave-one-field-out version gives the same results (CRPS ratio 1.000).
"""

POINT_PRIORS = {
    "SOC": {"mu_r": -1.9420396371535595, "s_r": 0.5911673116207268, "mu_s": -1.357009754749389, "s_s": 3.1428813937677367},
    "Clay": {"mu_r": -2.2426602538015903, "s_r": 0.519290372323507, "mu_s": -3.119837594291373, "s_s": 2.620453387184966},
    "pH": {"mu_r": -2.0546391522507914, "s_r": 0.6930469586482894, "mu_s": 0.0682144970697674, "s_s": 1.4201735699759765},
}

BLOCK_PRIOR = {"mu_r": -2.25, "s_r": 2.14, "mu_s": -0.32, "s_s": 1.82}   # 14 development fields

# Organic matter uses the SOC prior (Oxford, Meuse and the LimeSoDa fields that report organic matter).
ALIASES = {"soc": "SOC", "som": "SOC", "organic_carbon": "SOC", "organic_matter": "SOC", "om": "SOC", "clay": "Clay", "ph": "pH"}


def point_prior(target_kind):
    """Point-model prior for a property ('SOC', 'Clay', 'pH' or an alias). Other properties are not validated, so there is no default:
    pass your own prior dict to TabPFNSpatialGP(point_prior=...) if you need one."""
    key = ALIASES.get(str(target_kind).lower(), target_kind)
    if key not in POINT_PRIORS:
        raise ValueError(f"no validated point prior for target_kind={target_kind!r}; use one of {sorted(POINT_PRIORS)} "
                         "or pass point_prior={'mu_r': ..., 's_r': ..., 'mu_s': ..., 's_s': ...}")
    return dict(POINT_PRIORS[key])
