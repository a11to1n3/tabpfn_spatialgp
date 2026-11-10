"""tabpfn_spatialgp: TabPFN with a spatial Gaussian process on its out-of-fold errors, for field-scale soil mapping.

Point predictions come from TabPFN (with coordinates as inputs), corrected by the GP from 25 training points. Field and zone means get
uncertainty that accounts for the spatial correlation of errors (fitted from 3 training points, evaluated from 5). Built with TabPFN.
"""
from .model import TabPFNSpatialGP
from .estimator import TabPFNGPRegressor
from .bars import TabPFNBars, GaussianMixture
from .gp_field import ResidualField, field_extent
from .blocks import zone_blocks, block_predictive, as_index_blocks
from .priors import POINT_PRIORS, BLOCK_PRIOR

__all__ = ["TabPFNSpatialGP", "TabPFNGPRegressor", "TabPFNBars", "GaussianMixture", "ResidualField", "field_extent", "zone_blocks", "block_predictive",
           "as_index_blocks", "POINT_PRIORS", "BLOCK_PRIOR"]
__version__ = "0.0.1"
