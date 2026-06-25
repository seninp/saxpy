"""Testing brute force discord search implementation."""

import random

import numpy as np

from saxpy.discord import find_discords_brute_force


def _tie_heavy_series():
    """A periodic series whose aligned windows are identical (distance-0 ties),
    with one injected anomaly so a clear discord region also exists."""
    pattern = np.random.RandomState(0).randn(20)
    series = np.tile(pattern, 15)
    series[150:160] += 6.0
    return series


def test_brute_force():
    """Test brute force discord discovery."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=",")
    discords = find_discords_brute_force(dd[0:400], 100, 4)
    assert 3 == len(discords)
    assert 173 == discords[0][0]


def test_discord_distance_precision():
    """Pin the discord distances on real data so a refactor that perturbs the
    distance computation (znorm/early-abandon/min) is caught. Values verified
    against the reference run."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=",")
    discords = find_discords_brute_force(dd[100:500], 100, 2)
    assert [idx for idx, _ in discords] == [73, 219]
    np.testing.assert_allclose(
        [dist for _, dist in discords],
        [6.198555329625453, 5.563692399101614],
        rtol=0,
        atol=1e-9,
    )


def test_tie_break_is_deterministic_across_seeds():
    """Regression (audit #16): the unseeded VisitRegistry RNG used to make the
    winning discord index nondeterministic whenever two windows shared the
    maximal nearest-neighbour distance (first-visited-of-the-tied won). The
    deterministic lowest-index tie-break must pin the result regardless of the
    global RNG state -- here the same series yielded 100+ distinct winners
    before the fix."""
    series = _tie_heavy_series()
    results = set()
    for seed in range(8):
        random.seed(seed)
        np.random.seed(seed)
        discords = find_discords_brute_force(series, win_size=20, num_discords=1)
        results.add(tuple((idx, round(float(dist), 9)) for idx, dist in discords))
    assert len(results) == 1


def test_tie_break_prefers_lowest_index():
    """With every window identical (all NN distances 0.0), the deterministic
    tie-break must return the lowest index, not a random one."""
    series = np.tile(np.random.RandomState(0).randn(20), 15)
    for seed in (0, 7, 99):
        random.seed(seed)
        np.random.seed(seed)
        discords = find_discords_brute_force(series, win_size=20, num_discords=1)
        assert discords[0][0] == 0


def test_no_infinite_distance_discords():
    """Regression (audit #3): when the window is too large for any two
    non-overlapping subsequences to exist (max index gap < win_size), every
    nearest-neighbour distance stays +inf and there is no real discord. The old
    guard `~(np.inf == nn_distance)` was a bitwise-NOT on a Python bool (always
    truthy), so those +inf candidates slipped through as bogus discords. The fix
    `nn_distance < np.inf` must drop them, yielding an empty result.
    """
    series = np.random.RandomState(3).randn(50)
    discords = find_discords_brute_force(series, win_size=30, num_discords=1)
    assert discords == []
