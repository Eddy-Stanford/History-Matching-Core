from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from pyDOE import lhs

from . import stats


class SampleSpace:
    DEFAULT_RESOLUTION = 1000  # Default resolution for axis

    @classmethod
    def check_bounds(cls, bounds):
        for b in bounds:
            if len(b) > 2:
                raise ValueError("More than 2 bounds specified")
            if b[0] >= b[1]:
                raise ValueError("Lower bound MUST be lower than upper bound")

    @classmethod
    def from_bounds(cls, bounds, coord_labels=None):
        cls.check_bounds(bounds)
        ndims = len(bounds)
        if coord_labels is None:
            coord_labels = [str(i) for i in range(len(bounds))]
        sample_space = xr.DataArray(
            data=np.ones([2 for _ in range(ndims)]),
            dims=coord_labels,
            coords={l: list(bound) for l, bound in zip(coord_labels, bounds)},
        )
        instance = cls(bounds=bounds, sample_space=sample_space)
        return instance

    @classmethod
    def from_bounds_dict(cls, bounds_dict):
        return cls.from_bounds(bounds_dict.values(), coord_labels=bounds_dict.keys())

    @classmethod
    def from_xarray(cls, xr_ds: xr.DataArray):
        xr_ds = xr_ds.astype(float)
        bounded = np.nonzero(xr_ds.data)
        bounds = [
            (float(xr_ds[coord][np.min(idxs)]), float(xr_ds[coord][np.max(idxs)]))
            for idxs, coord in zip(bounded, xr_ds.coords)
        ]
        cropped_ds = xr_ds.sel(
            {k: slice(v[0], v[1]) for k, v in zip(xr_ds.dims, bounds)}
        )
        instance = cls(
            bounds=bounds,
            sample_space=cropped_ds,
        )
        return instance

    @classmethod
    def from_numpy(
        cls,
        space: np.ndarray,
        **coordinates,
    ):
        ds = xr.DataArray(
            space.astype(float), dims=coordinates.keys(), coords=coordinates
        )
        return cls.from_xarray(ds)

    def __init__(
        self,
        bounds: Sequence[Sequence[float]],
        sample_space: xr.DataArray,
    ):
        self._bounds = bounds
        self._sample_space = sample_space
        self._bounds_dict = None

    @property
    def ndims(self):
        return len(self.coord_labels)

    @property
    def coord_labels(self):
        return self._sample_space.dims

    @property
    def bounds_dict(self):
        if not self._bounds_dict:
            self._bounds_dict = {
                label: bounds
                for label, bounds in zip(self._sample_space.dims, self._bounds)
            }
        return self._bounds_dict

    @property
    def is_square_space(self):
        return (self._sample_space > 0.5).all()

    def __rescale_samples(self, samples):
        """
        Rescale samples in 0-1 range uniformly to match bounding box of space.
        Cheap'n'easy
        """
        samples = np.column_stack((samples, np.ones(samples.shape[0])))
        ranges = [int_max - int_min for (int_min, int_max) in self._bounds]
        mins = [int_min for (int_min, _) in self._bounds]
        rescale = np.diag(ranges)
        rescale = np.append(rescale, [mins], axis=0)
        return samples @ rescale

    def __uniform_lhs_sample(self, n_samples, criterion=None):
        # TODO: Consider moving this to stats.py?
        samples = lhs(self.ndims, samples=n_samples, criterion=criterion)
        return self.__rescale_samples(samples)

    def centroid(self):
        sxr = self.to_xarray().astype(int)
        denom = sxr.integrate(coord=self.coord_labels)
        return np.array(
            [
                (sxr[c] * sxr).integrate(coord=self.coord_labels) / denom
                for c in self.coord_labels
            ]
        )

    def lhs_sample(self, n_samples, criterion=None, labelled=False):
        if self.is_square_space:
            samples = self.__uniform_lhs_sample(n_samples, criterion=criterion)
        else:
            samples = lhs(self.ndims, n_samples, criterion=None)
            norm = self._sample_space
            for d in self.coord_labels:
                norm = norm.integrate(d)
            pdf = self._sample_space / norm
            samples = stats.inverse_transform_samples(samples, pdf)
        if labelled:
            return pd.DataFrame(samples, columns=self.coord_labels)
        return samples

    def uniform(self, n_samples, labelled=False):
        if self.is_square_space:
            ## Cheap method just rescale
            samples = np.random.uniform(size=(n_samples, self.ndims))
            samples = self.__rescale_samples(samples)
        else:
            samples = []
            while len(samples) < n_samples:
                point = np.random.uniform(
                    low=[b[0] for b in self._bounds], high=[b[1] for b in self._bounds]
                )
                if self.inspace(**{k: v for k, v in zip(self.coord_labels, point)}):
                    samples.append(point)

            samples = np.array(samples)
        if labelled:
            return pd.DataFrame(samples, columns=self.coord_labels)
        return samples

    def inspace(self, **r) -> bool:
        if self._sample_space is None:
            # Uniform sample space
            for k, v in r.items():
                limit_min, limit_max = self.bounds_dict[k]
                if limit_min > v or v > limit_max:
                    return False
            return True
        else:
            return bool(self._sample_space.interp(**r) > 0.5)

    def to_xarray(self, resolution=None):
        resolution = self.DEFAULT_RESOLUTION if resolution is None else resolution
        return (
            self._sample_space.interp(
                coords={
                    k: np.linspace(v[0], v[1], resolution)
                    for k, v in self.bounds_dict.items()
                }
            )
            > 0.5
        )

    def intersection(self, other: "SampleSpace"):
        if self.coord_labels != other.coord_labels:
            raise ValueError("Conflicting Sample Spaces")
        new_bound_dict = {
            k: [
                max(self.bounds_dict[k][0], other.bounds_dict[k][0]),
                min(self.bounds_dict[k][1], other.bounds_dict[k][1]),
            ]
            for k in self.coord_labels
        }
        self_smpl = (
            self._sample_space.interp(
                coords={
                    k: np.linspace(v[0], v[1], self.DEFAULT_RESOLUTION)
                    for k, v in new_bound_dict.items()
                }
            )
            > 0.5
        )
        other_smpl = (
            other.to_xarray()
            .astype(float)
            .interp(
                coords={
                    k: np.linspace(v[0], v[1], self.DEFAULT_RESOLUTION)
                    for k, v in new_bound_dict.items()
                }
            )
            > 0.5
        )

        return self.from_xarray(self_smpl & other_smpl)

    def __and__(self, __t):
        return self.intersection(__t)

    def union(self, other: "SampleSpace"):
        if self.coord_labels != other.coord_labels:
            raise ValueError("Conflicting Sample Spaces")
        new_bound_dict = {
            k: [
                min(self.bounds_dict[k][0], other.bounds_dict[k][0]),
                max(self.bounds_dict[k][1], other.bounds_dict[k][1]),
            ]
            for k in self.coord_labels
        }
        self_smpl = (
            self._sample_space.interp(
                coords={
                    k: np.linspace(v[0], v[1], self.DEFAULT_RESOLUTION)
                    for k, v in new_bound_dict.items()
                }
            )
            > 0.5
        )
        other_smpl = (
            other.to_xarray()
            .astype(float)
            .interp(
                coords={
                    k: np.linspace(v[0], v[1], self.DEFAULT_RESOLUTION)
                    for k, v in new_bound_dict.items()
                }
            )
            > 0.5
        )

        return self.from_xarray(self_smpl | other_smpl)

    def __or__(self, __t):
        return self.union(__t)
