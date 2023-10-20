import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import MinMaxScaler

from ..samples import SampleSpace


class Emulator(BaseEstimator):
    def __init__(
        self, n_features=1, random_state=None, kernel=None, n_restarts_optimizer=0
    ) -> None:
        self.random_state = random_state

        self.n_features = n_features
        if kernel is None:
            self.kernel = (
                ConstantKernel() * RBF(length_scale=np.ones((self.n_features,)))
                + WhiteKernel()
            )
        else:
            self.kernel = kernel
        self.scaler_x = MinMaxScaler()
        self.n_restarts_optimizer = n_restarts_optimizer
        self.__gps = [
            GaussianProcessRegressor(
                normalize_y=True,
                random_state=self.random_state,
                kernel=self.kernel,
                n_restarts_optimizer=self.n_restarts_optimizer,
            )
            for _ in range(self.n_features)
        ]
        super().__init__()

    def fit(self, X, y):
        X = self.scaler_x.fit_transform(X)
        for i, gp in enumerate(self.__gps):
            gp.fit(X, y[:, i])

    def predict(self, X, return_std=False):
        X = self.scaler_x.transform(X)
        values = np.array([gp.predict(X, return_std=return_std) for gp in self.__gps])
        if return_std:
            return np.transpose(values[:, 0]), np.transpose(values[:, 1])
        return np.transpose(values)

    def predict_over_space(self, space: SampleSpace, return_std=False, resolution=None):
        space_xr = space.to_xarray(resolution=resolution)
        valid_points = np.array(np.nonzero(space_xr.values))
        X = np.zeros(valid_points.shape)
        for i, dim in enumerate(space_xr.dims):
            X[i, :] = space_xr[dim][valid_points[i, :]]
        X = np.transpose(X)
        dims = ("n_features", *space_xr.dims)
        coords = {"n_features": np.arange(self.n_features), **space_xr.coords}
        predictions = np.empty((self.n_features, *space_xr.shape)) * np.nan
        if return_std:
            predictions_std = np.empty((self.n_features, *space_xr.shape)) * np.nan
            preds_flt, preds_std_flt = self.predict(X, return_std=True)
            for pred, pred_std, valid_point in zip(
                preds_flt, preds_std_flt, np.transpose(valid_points)
            ):
                predictions[(slice(None), *valid_point)] = pred
                predictions_std[(slice(None), *valid_point)] = pred_std

            return xr.DataArray(predictions, dims=dims, coords=coords), xr.DataArray(
                predictions_std, dims=dims, coords=coords
            )
        else:
            preds_flt = self.predict(X, return_std=False)
            for pred, valid_point in zip(preds_flt, valid_points):
                predictions[(slice(None), *valid_point)] = pred
            return xr.DataArray(predictions, dims=dims, coords=coords)
