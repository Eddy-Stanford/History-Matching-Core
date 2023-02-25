from functools import partial

import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid


def conditional_cumulative_prob(pdf: xr.DataArray, dim: str) -> xr.DataArray:
    axis = pdf.get_axis_num(dim)
    jun_norm = xr.apply_ufunc(
        partial(cumulative_trapezoid, axis=axis, initial=0), pdf, pdf[dim]
    )
    normalizer = pdf.integrate(dim)
    return jun_norm / normalizer


def get_conditional_cumulative(pdf, new_dim_name="d"):
    return xr.concat(
        [conditional_cumulative_prob(pdf, l) for l in pdf.dims],
        new_dim_name,
    )


def inverse_transform_samples(uniform_samples, pdf):
    Jxyz = get_conditional_cumulative(pdf).expand_dims("n_samples")
    ndims = Jxyz.shape[1]
    residual = (
        Jxyz
        - uniform_samples.reshape((*uniform_samples.shape, *[1 for _ in range(ndims)]))
    ) ** 2
    residual = residual.sum(dim="d")
    point_loc = residual.argmin(dim=pdf.dims)
    return np.transpose(np.array([pdf[k][v] for k, v in point_loc.items()]))
