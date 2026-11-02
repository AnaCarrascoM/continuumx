"""The upper-envelope routine against a direct maximum over a fine grid."""

import random

import pytest

from continuumx.structures.piecewise import HatFunction, PiecewiseLinear, upper_envelope

GRID = [i / 500 for i in range(501)]


def _random_hats(seed: int, count: int, slope: float = 2.0):
    rng = random.Random(seed)
    hats = []
    for _ in range(count):
        a, b = sorted((rng.random(), rng.random()))
        hats.append(HatFunction(a, b, rng.uniform(0.0, 5.0), slope))
    return hats


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("count", [1, 2, 5, 17])
def test_envelope_matches_pointwise_maximum(seed, count):
    hats = _random_hats(seed, count)
    envelope = upper_envelope(hats)
    for theta in GRID:
        expected = max(hat(theta) for hat in hats)
        assert envelope(theta) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_envelope_maximum_is_the_largest_ceiling(seed):
    hats = _random_hats(seed, 9)
    envelope = upper_envelope(hats)
    assert envelope.maximum() == pytest.approx(max(hat.h for hat in hats))


@pytest.mark.parametrize("seed", range(10))
def test_envelope_minimum_is_attained_where_reported(seed):
    hats = _random_hats(seed, 9)
    envelope = upper_envelope(hats)
    value, intervals = envelope.minimum()
    assert value == pytest.approx(min(envelope(t) for t in GRID), abs=1e-2)
    for lo, hi in intervals:
        assert envelope(lo) == pytest.approx(value)
        assert envelope(hi) == pytest.approx(value)
        assert envelope((lo + hi) / 2) == pytest.approx(value)


def test_degenerate_hats_reaching_the_endpoints():
    hats = [HatFunction(0.0, 0.3, 2.0, 1.0), HatFunction(0.7, 1.0, 2.0, 1.0)]
    envelope = upper_envelope(hats)
    # This is the "valley" of the same-edge eccentricity.
    for theta in GRID:
        assert envelope(theta) == pytest.approx(
            max(h(theta) for h in hats), abs=1e-9
        )
    # The minimum sits midway between the two intervals [b, s].
    value, intervals = envelope.minimum()
    assert value == pytest.approx(2.0 - 0.2)
    assert intervals == [pytest.approx((0.5, 0.5))]


def test_envelope_requires_a_common_slope():
    with pytest.raises(ValueError):
        upper_envelope([HatFunction(0.0, 1.0, 1.0, 1.0),
                        HatFunction(0.0, 1.0, 1.0, 2.0)])


def test_piecewise_linear_rejects_decreasing_abscissas():
    with pytest.raises(ValueError):
        PiecewiseLinear([0.0, 0.5, 0.2], [0.0, 1.0, 2.0])


def test_minimum_reports_a_flat_interval():
    f = PiecewiseLinear([0.0, 0.25, 0.75, 1.0], [1.0, 0.0, 0.0, 1.0])
    value, intervals = f.minimum()
    assert value == 0.0
    assert intervals == [(0.25, 0.75)]
