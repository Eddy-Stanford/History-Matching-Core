import numpy as np
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler


def implausibility(
    predict_mean: np.ndarray,
    predict_err: np.ndarray,
    obs_mean: np.ndarray,
    obs_err: np.ndarray,
):
    """
    Calculate implausability score for set of predictions

    Input:
    predict_mean (ndarray) - (n_features,n_samples) Array of predictions
    obs_mean (ndarray) - (n_features) Mean of observational data
    predict_err - Standard Error in predictoins
    obs_err- Standard error in mean of observations
    """
    return np.sqrt(
        np.abs(predict_mean - obs_mean[:, None])
        / np.sqrt(predict_err[:, None] ** 2 + obs_err**2)
    )


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
            return (values[:, 0, :] * self.__std_y) + self.__mean_y, values[
                :, 1, :
            ] * self.__std_y
        return values * self.__std_y + self.__mean_y
