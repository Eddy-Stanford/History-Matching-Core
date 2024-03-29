import numpy as np
import pytest
import xarray

from history_matching.samples import SampleSpace


@pytest.mark.parametrize(
    "bounds_dict,", [{"0": (1, 2)}, {"0": (1, 2), "1": (10, 20), "2": (-10, 10)}]
)
def test_from_bounds_dict(bounds_dict):
    space = SampleSpace.from_bounds_dict(bounds_dict)
    assert isinstance(space, SampleSpace)
    assert space.ndims == len(bounds_dict)
    assert space.bounds_dict == bounds_dict


@pytest.mark.parametrize(
    "bounds,",
    [
        [
            (1, 2),
        ],
        [(1, 2), (2, 3)],
        [(1, 2), (2, 3), (-10, 10)],
        [(1, 2), (2, 3), (-10, 10), (-5, 10423)],
    ],
)
def test_from_bounds(bounds):
    space = SampleSpace.from_bounds(bounds)
    assert isinstance(space, SampleSpace)
    assert space.ndims == len(bounds)
    for i, b in enumerate(bounds):
        assert space.bounds_dict[str(i)] == b


def test_wrong_bound_order():
    with pytest.raises(ValueError) as e:
        space = SampleSpace.from_bounds([(1, 1)])


def test_incorrect_bounds():
    with pytest.raises(ValueError) as e:
        space = SampleSpace.from_bounds([(1, 2, 3)])


@pytest.mark.parametrize(
    "bounds,",
    [
        [
            (1, 2),
        ],
        [(1, 2), (2, 3)],
        [(1, 2), (2, 3), (-10, 10)],
        [(1, 2), (2, 3), (-10, 10), (-5, 10423)],
    ],
)
def test_lhs_space(bounds):
    space = SampleSpace.from_bounds(bounds)
    sample = space.lhs_sample(5, "c")
    assert sample.shape == (5, space.ndims)
    for dim in range(space.ndims):
        assert (sample[:, dim] < bounds[dim][1]).all()
        assert (sample[:, dim] >= bounds[dim][0]).all()


@pytest.mark.parametrize(
    "bounds,",
    [
        [
            (1, 2),
        ],
        [(1, 2), (2, 3)],
        [(1, 2), (2, 3), (-10, 10)],
        [(1, 2), (2, 3), (-10, 10), (-5, 10423)],
    ],
)
def test_uniform_space(bounds):
    space = SampleSpace.from_bounds(bounds)
    sample = space.uniform(5)
    assert sample.shape == (5, space.ndims)
    for dim in range(space.ndims):
        assert (sample[:, dim] < bounds[dim][1]).all()
        assert (sample[:, dim] >= bounds[dim][0]).all()


@pytest.mark.parametrize(
    "dim,",
    [1, 2, 3, 4, 5],
)
def test_from_xarray(dim):
    data = (
        np.sum(np.meshgrid(*[np.linspace(0, 1, 10) for _ in range(dim)]), axis=0) / dim
        > 0.5
    )
    ds = xarray.DataArray(
        data=data, coords={str(d): np.linspace(0, 1, 10) for d in range(dim)}
    )
    space = SampleSpace.from_xarray(ds)
    assert isinstance(space, SampleSpace)
    samples = space.uniform(100)
    assert (samples.sum(axis=-1) > 0.5).all()  # Check samples make sense
