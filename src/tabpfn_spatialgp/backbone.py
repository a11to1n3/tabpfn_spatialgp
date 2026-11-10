"""TabPFN part: TabPFN v2 with its original weights, 4 ensemble members and no target transform.

Inputs are the covariates followed by the two projected coordinates. TabPFN is never fine-tuned: during the method's development, fine-tuning
(on real fields or on synthetic GP tasks) added nothing.
"""
from . import settings as S_


def make_regressor(seed=S_.RANDOM_STATE, device="auto", model_path="auto"):
    """A fresh, unfitted TabPFN regressor with the package's settings. device='auto' uses the GPU if there is one, otherwise the CPU;
    model_path='auto' uses TabPFN's default v2 regressor checkpoint (downloaded on first use)."""
    from importlib.metadata import version
    from tabpfn import TabPFNRegressor
    if version("tabpfn") != S_.TABPFN_VERSION:
        import warnings
        warnings.warn(f"tabpfn {version('tabpfn')} is installed; the paper's results and kappa were obtained with tabpfn {S_.TABPFN_VERSION}")
    return TabPFNRegressor(model_path=model_path, device=device, n_estimators=S_.N_ESTIMATORS, random_state=seed,
                           inference_config=S_.INFERENCE_CONFIG)
