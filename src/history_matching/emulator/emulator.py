import numpy as np
import xarray as xr
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from ..samples import SampleSpace


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
    predict_mean: xr.DataArray,
    predict_err: xr.DataArray,
    obs_mean: np.ndarray,
    obs_err: np.ndarray,
) -> xr.DataArray:
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
    return xr.concat(
        [
            (predict_mean[i] - obs_mean[i]) ** 2
            / (predict_err[i] ** 2 + obs_err[i] ** 2)
            for i in range(n_features)
        ],
        dim="n_features",
    ).sum(dim="n_features")


def implausibility(
    predict_mean: xr.DataArray,
    predict_err: xr.DataArray,
    obs_mean: np.ndarray,
    obs_err: np.ndarray,
) -> xr.DataArray:
    return np.sqrt(implausibility2(predict_mean, predict_err, obs_mean, obs_err))


def chisquaredtest(imp: xr.DataArray, significance: float):
    """
    imp: Implausibility DataArray, must be squared variation
    significance: 1- Significance level of chi squared test, e.g 95% SL would input 5% here.
    """
    return imp < chi2.isf(significance, len(imp.dims))


class GPEmulator(BaseEstimator):
    def __init__(self, n_features=2, random_state=None, kernel=None):
        self.random_state = random_state
        self.kernel = kernel
        self.n_features = n_features
        self.gps = None
        self.scaler = StandardScaler()
        self.__mean_y = None
        self.__std_y = None

    def fit(self, X, y, y_err):
        if X.shape[-1] != self.n_features:
            raise ValueError(
                "Last dimension of x do not agree with the specified n_features "
            )
        if len(y.shape) == 1:
            y = y[:, None]
        if len(y_err.shape) == 1:
            y_err = y_err[:, None]
        self.__mean_y = np.mean(y, axis=0)
        self.__std_y = np.std(y, axis=0)
        y = (y - self.__mean_y) / self.__std_y
        y_err = y_err / self.__std_y

        self.scaler.fit(X)
        X = self.scaler.transform(X)

        self.gps = [
            GaussianProcessRegressor(
                alpha=y_err[:, i] ** 2,
                kernel=self.kernel,
                random_state=self.random_state,
            )
            for i in range(self.n_features)
        ]
        for i, gp in enumerate(self.gps):
            gp.fit(X, y[:, i])

    def predict(self, X, return_std=False):
        X = self.scaler.transform(X)
        values = np.array([gp.predict(X, return_std=return_std) for gp in self.gps])
        if return_std:
            return np.transpose(
                (values[:, 0, :] * self.__std_y[:, None]) + self.__mean_y[:, None]
            ), np.transpose(values[:, 1, :] * self.__std_y[:, None])
        return np.transpose(values * self.__std_y[:, None] + self.__mean_y[:, None])

    def predict_over_space(self, space: SampleSpace, return_std=False, resolution=None):
        """
        Run prediction over sample space
        """
        space_up = space.to_xarray(resolution=resolution)
        valid_points = np.array(np.nonzero(space_up.values))
        X = np.zeros(valid_points.shape)
        for i, dim in enumerate(space_up.dims):
            X[i, :] = space_up[dim][valid_points[i, :]]
        X = np.transpose(X)

        # TODO: This can almost certainly be implemented better with less code duplication
        if return_std:
            pred, pred_std = self.predict(X, return_std=True)
            predictions = np.empty((self.n_features, *space_up.shape)) * np.nan
            predictions_std = np.empty((self.n_features, *space_up.shape)) * np.nan
            for pred_point, pred_std, valid_point in zip(
                pred, pred_std, valid_points.T
            ):
                predictions[(slice(None), *valid_point)] = pred_point
                predictions_std[(slice(None), *valid_point)] = pred_std
            return xr.DataArray(
                predictions,
                dims=("n_features", *space_up.dims),
                coords={"n_features": np.arange(self.n_features), **space_up.coords},
            ), xr.DataArray(
                predictions_std,
                dims=("n_features", *space_up.dims),
                coords={"n_features": np.arange(self.n_features), **space_up.coords},
            )

        else:
            pred = self.predict(X, return_std=False)
            predictions = np.empty((self.n_features, *space_up.shape)) * np.nan
            for pred_point, valid_point in zip(pred, valid_points.T):
                predictions[(slice(None), *valid_point)] = pred_point
            return xr.DataArray(
                predictions,
                dims=("n_features", *space_up.dims),
                coords={"n_features": np.arange(self.n_features), **space_up.coords},
            )
