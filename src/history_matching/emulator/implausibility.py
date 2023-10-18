from typing import List, Union

import numpy as np
import xarray as xr
from scipy.stats import chi2


def implausibility_inf(
    predict_mean: xr.DataArray,
    predict_err: xr.DataArray,
    obs_mean: np.ndarray,
    obs_err: np.ndarray,
    sqrt: bool = True,
) -> xr.DataArray:
    n_features = obs_mean.shape[0]
    if n_features != predict_mean.shape[0]:
        raise ValueError("Incompatible shapes between observations and predictions")
    return xr.concat(
        [
            (predict_mean[i] - obs_mean[i]) ** 2
            / (predict_err[i] ** 2 + obs_err[i] ** 2)
            for i in range(n_features)
        ],
        dim="n_features",
    ).max(dim="n_features") ** (0.5 if sqrt else 1)


def implausibility2(
    predict_mean: Union[np.ndarray, xr.DataArray],
    predict_err: Union[np.ndarray, xr.DataArray],
    obs_mean: Union[np.ndarray, List],
    obs_err: Union[np.ndarray, List],
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate implausibility score for set of predictions

    Input:
    predict_mean (ndarray) - (n_features,n_samples) Array of predictions
    obs_mean (ndarray) - (n_features) Mean of observational data
    predict_err - Standard Error in predictions
    obs_err- Standard error in mean of observations
    """
    n_features = obs_mean.shape[0]
    if n_features != predict_mean.shape[0]:
        raise ValueError("Incompatible shapes between observations and predictions")
    if isinstance(predict_mean, xr.DataArray):
        return xr.concat(
            [
                (predict_mean[i] - obs_mean[i]) ** 2
                / (predict_err[i] ** 2 + obs_err[i] ** 2)
                for i in range(n_features)
            ],
            dim="n_features",
        ).sum(dim="n_features")
    if isinstance(predict_mean, np.ndarray):
        return np.stack(
            [
                (predict_mean[i] - obs_mean[i]) ** 2
                / (predict_err[i] ** 2 + obs_err[i] ** 2)
                for i in range(n_features)
            ]
        ).sum(axis=0)


def likilihood(
    predict_mean: Union[np.ndarray, xr.DataArray],
    predict_err: Union[np.ndarray, xr.DataArray],
    obs_mean: Union[np.ndarray, List],
    obs_err: Union[np.ndarray, List],
) -> Union[np.ndarray, xr.DataArray]:
    return np.exp(-0.5 * implausibility2(predict_mean, predict_err, obs_mean, obs_err))


def implausibility(
    predict_mean: Union[np.ndarray, xr.DataArray],
    predict_err: Union[np.ndarray, xr.DataArray],
    obs_mean: Union[np.ndarray, List],
    obs_err: Union[np.ndarray, List],
) -> Union[np.ndarray, xr.DataArray]:
    return np.sqrt(implausibility2(predict_mean, predict_err, obs_mean, obs_err))


def chisquaredtest(imp: xr.DataArray, significance: float):
    """
    imp: Implausibility DataArray, must be squared variation
    significance: 1- Significance level of chi squared test, e.g 95% SL would input 5% here.
    """
    return imp < chi2.isf(significance, len(imp.dims))
