"""Testing the RRA (Rare Rule Anomaly) discord discovery implementation.

The reference C++ discord region (start=398, end=499 on ecg0606, covering the
known anomaly near position 420) is the golden target. RRA's grammar rule
numbering and random search order are not portable, so the tests assert the
structural result -- the top discord covers the anomaly region, and RRA agrees
with HOT-SAX on that region -- rather than exact start/end indices.
"""

import os

import numpy as np
import pytest

from saxpy.hotsax import find_discords_hotsax
from saxpy.rra import RRADiscord, _normalized_distance, _paa2, find_discords_rra

ECG = os.path.join(os.path.dirname(__file__), os.pardir, "data", "ecg0606_1.csv")


def _ecg():
    return np.genfromtxt(ECG, delimiter=",")


def test_rra_finds_ecg_anomaly():
    """The top discord must cover the known ecg0606 anomaly near position 420
    (mirrors the jmotif test_discord_rra.R and the saxpy HOT-SAX test)."""
    discords = find_discords_rra(_ecg(), 100, 4, 4, num_discords=4)
    assert len(discords) >= 1
    top = discords[0]
    assert top.start < 420 < top.end


def test_rra_agrees_with_hotsax_region():
    """RRA and HOT-SAX must identify the same primary anomaly region -- an
    independent cross-check between two different discord algorithms."""
    ecg = _ecg()
    rra = find_discords_rra(ecg, 100, 4, 4, num_discords=1)
    hot = find_discords_hotsax(ecg, win_size=100, num_discords=1, paa_size=4, alphabet_size=4)
    assert rra[0].start < hot[0][0] < rra[0].end


def test_rra_is_deterministic():
    """Seeded internally, so repeated calls give identical discords."""
    ecg = _ecg()
    a = find_discords_rra(ecg, 100, 4, 4, num_discords=3)
    b = find_discords_rra(ecg, 100, 4, 4, num_discords=3)
    assert [(d.start, d.end) for d in a] == [(d.start, d.end) for d in b]


def test_rra_discords_are_disjoint():
    """Reported discords must not overlap (mutual exclusion via global visited)."""
    discords = find_discords_rra(_ecg(), 100, 4, 4, num_discords=4)
    spans = sorted((d.start, d.end) for d in discords)
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 <= s2  # no overlap


def test_rra_returns_rradiscord_records():
    discords = find_discords_rra(_ecg(), 100, 4, 4, num_discords=1)
    assert isinstance(discords[0], RRADiscord)
    d = discords[0]
    assert d.length == d.end - d.start
    assert d.nn_distance > 0


def test_rra_rejects_window_too_large():
    with pytest.raises(ValueError):
        find_discords_rra(np.zeros(50), win_size=100)


def test_paa2_matches_reference():
    """_paa2 fractional-boundary PAA, pinned against jmotif C++ output."""
    np.testing.assert_allclose(
        _paa2([1, 2, 3, 4, 5, 6, 7], 3),
        [1.7142857142857142, 4.0, 6.285714285714286],
        rtol=0,
        atol=1e-12,
    )
    # paa_num == len is the identity.
    np.testing.assert_array_equal(_paa2([3.0, 1.0, 4.0], 3), [3.0, 1.0, 4.0])


def test_paa2_equals_saxpy_paa():
    """_paa2 IS saxpy.paa -- both are overlap-weighted fractional-boundary PAA.

    RRA's _paa2 used to carry its own loop (a port of the C++ _paa2) that, before
    the C++ snapped the final break to ``len``, dropped the last sample when
    ``len/paa_num`` rounded just past an integer. That bug was the *only* reason
    _paa2 ever differed from saxpy.paa; the predecessor test pinned the bug. _paa2
    now delegates to saxpy.paa, so this pins that they agree, including on the
    non-divisible 29->7 case whose mid-sample boundaries exercise the fraction.
    """
    from saxpy.paa import paa

    series = np.random.RandomState(1).randn(29)
    np.testing.assert_allclose(_paa2(list(series), 7), paa(list(series), 7), atol=1e-12)
    # Pinned against jmotif's C++ paa() (== _paa2 with the boundary fix).
    assert _paa2(list(series), 7)[-1] == pytest.approx(-0.454175, abs=1e-5)


def test_normalized_distance_equal_length():
    """Equal-length spans: sqrt(sum of squared diffs) / count."""
    series = np.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
    # compare [0:2]=(0,0) with [3:5]=(3,4): dist = sqrt(9+16)/2 = 5/2
    assert _normalized_distance(0, 2, 3, 5, series) == pytest.approx(2.5)


def test_normalized_distance_is_symmetric_in_length_handling():
    """Unequal lengths use _paa2 on the longer; result is finite and >= 0."""
    series = np.linspace(0, 1, 20)
    d = _normalized_distance(0, 4, 5, 13, series)
    assert d >= 0 and np.isfinite(d)
