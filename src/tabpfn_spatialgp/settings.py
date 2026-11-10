"""Settings of the method.

These are the values behind the results of the paper. Changing any of them gives a different model, one that has not been
validated.
"""
import numpy as np

# --- TabPFN part ---------------------------------------------------------------------------------------------------------------------
TABPFN_VERSION = "2.0.9"            # package version behind every result; released (not fine-tuned) weights
N_ESTIMATORS = 4                    # TabPFN's default is 8; 4 gives the same accuracy (development: 1.002 times the squared error) at half the cost
INFERENCE_CONFIG = {"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,)}   # no target power transform (development: 0.946 times the squared error, 5-30 % revealed)
RANDOM_STATE = 0                    # TabPFN seed and the seed of the out-of-fold split
N_SPECTRAL_PCS = 10                 # spectral covariates -> 10 principal components (n - 1 when n <= 10), fitted on the training rows
BLAS_THREADS = 1                    # TabPFN's output depends on the number of BLAS/OpenMP threads for some inputs (one test cell:
                                    # SP.231, n = 50, changed by up to 0.40, target sd 1.74, with 4 threads). The study's runs used
                                    # 1 thread, so fit and predict enforce it (threadpoolctl, with torch loaded first).

# --- Out-of-fold residuals -----------------------------------------------------------------------------------------------------------
OOF_FOLDS = 5                       # TabPFN is refitted 5 times; each training row is predicted by a model that did not see it

# --- Residual GP (shared by the point and the block model) ---------------------------------------------------------------------------
GRID_LOG_RANGE_OFFSETS = np.linspace(-1.5, 1.5, 7)    # averaging grid: ranges = MAP range x exp(offset)
GRID_SPATIAL_SHARES = (0.1, 0.25, 0.5, 0.75, 0.9)    # averaging grid: spatial share sigma^2 / (sigma^2 + tau^2); total variance fixed
VARIANCE_BOUNDS = (1e-4, 10.0)      # bounds of sigma^2 and tau^2 on the standardised residual scale
RANGE_UPPER_LOG_OFFSET = 4.0        # range <= extent * exp(prior mean of log(range / extent) + 4)
MAP_START_LOG_OFFSETS = (-0.7, 0.0, 0.7)              # L-BFGS-B starts: prior median range x exp(offset)
MAP_START_SIGMA2, MAP_START_TAU2 = 0.3, 0.7           # starting variances (standardised scale)
MAP_MAXITER = 300
JITTER = 1e-8                       # added to the diagonal of K for numerical stability

# --- Point predictions ---------------------------------------------------------------------------------------------------------------
POINT_GP_MIN_N = 25                 # the GP corrects point predictions only from 25 training points; below that, TabPFN alone
N_MIXTURE = 400                     # TabPFN's distribution is represented by 400 equally spaced quantiles for the convolution

# --- Field and zone means ------------------------------------------------------------------------------------------------------------
BLOCK_GP_MIN_N = 3                  # the block model's GP (population prior) is fitted from 3 training points (evaluated from 5)
BLOCK_MEAN_CORRECTION_MIN_N = 25    # the GP's block-mean correction is applied only from 25 training points
KAPPA = 1.15                        # factor on the block standard deviation, the study's value: the factor rule on the factor-setting
                                    # data with each task counted once
ZONE_K = 4                          # zones: k-means clusters of the field's sample locations ...
ZONE_MIN_POINTS = 3                 # ... with at least 3 prediction locations

Z80 = 1.2815515655446004            # standard-normal 90 % quantile: half-width of a central 80 % interval in sd units
