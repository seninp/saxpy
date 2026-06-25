"""Testing PAA implementation."""

import numpy as np
import pytest

from saxpy import paa


def test_paa1():
    """Test points to points via PAA."""
    y = np.array([-1, -2, -1, 0, 2, 1, 1, 0])
    v = np.array([-1.375, 0.75, 0.625])
    np.testing.assert_array_equal(paa.paa(y, 3), v)
    np.testing.assert_array_equal(paa.paa(y, 8), y)
    np.testing.assert_array_equal(paa.paa(y, 2), np.array([-1, 1]))


def test_paa2():
    """An irregular points to points via PAA (15 -> 10)."""
    dat = np.array(
        [
            -0.9796808,
            -0.8622706,
            -0.6123005,
            0.8496459,
            1.739691,
            1.588194,
            1.095829,
            0.5277147,
            0.4709033,
            -0.2865819,
            0.0921607,
            -0.2865819,
            -0.9039323,
            -1.195564,
            -1.237226,
        ]
    )
    dat_paa = np.array(
        [
            -0.9405441,
            -0.6956239,
            1.146328,
            1.638693,
            0.9064573,
            0.4898404,
            -0.1603344,
            -0.1603344,
            -1.001143,
            -1.223339,
        ]
    )
    np.testing.assert_array_almost_equal(paa.paa(dat, 10), dat_paa, decimal=6)


def test_paa_rejects_non_positive_segment_size():
    """Regression (audit #34): paa_size==0 used to raise a bare
    ZeroDivisionError (integer modulo by zero); negatives were undefined. PAA
    must reject non-positive segment sizes with a clear ValueError."""
    with pytest.raises(ValueError):
        paa.paa([1, 2, 3], 0)
    with pytest.raises(ValueError):
        paa.paa([1, 2, 3], -2)


def test_paa_rejects_empty_series():
    """Regression (audit #34): an empty series silently produced an all-NaN
    array that then propagated through the whole SAX pipeline. It must fail
    loudly instead."""
    with pytest.raises(ValueError):
        paa.paa([], 3)


def test_paa_rejects_upsampling():
    """Regression (audit #33): when paa_size > series length the divisor branch
    interpolated the series *up* (e.g. 3 points -> 5 segments) instead of
    reducing it. PAA only downsamples, so this must raise ValueError -- matching
    the paa_size > win_size guard already enforced in sax_via_window."""
    with pytest.raises(ValueError):
        paa.paa([1, 2, 3], 5)


def test_paa_rejects_multicolumn_for_unidim():
    """Audit follow-up: a 1-D sax_type ('unidim'/'zscore'/'independent') used to
    silently average only column 0 and drop the rest of a multi-column array
    (e.g. paa([[1,2],[3,4]], 1, 'unidim') -> [2.]). That is silent data loss; it
    must raise instead. Multi-dimensional input belongs to repeat/energy/
    independent. This also covers the last uncovered branch in paa.py."""
    with pytest.raises(ValueError, match="1-D series"):
        paa.paa([[1, 2], [3, 4]], 1, "unidim")

    # A single-column 2-D array is still fine (it is effectively 1-D).
    np.testing.assert_array_equal(paa.paa([[1.0], [3.0]], 1, "unidim"), np.array([2.0]))


def test_paa_multidim_energy_and_repeat():
    """Audit #40 follow-up: the np.add.at -> reshape/mean vectorization is
    exercised directly for the multi-dimensional 'energy'/'repeat' branches and
    the non-divisible path (only unidim and divisible doctests existed before).

    Expected values are derived independently from the PAA segment-mean formula
    (expand each column by paa_size, contract by series_len), not read back from
    the implementation. Each column is PAA'd separately and the result is
    (paa_size x num_dims).
    """
    series = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float
    )
    # Non-divisible: 5 rows -> 2 segments. Column 0 = [1,4,7,10,13]:
    #   seg0 = (2*1 + 2*4 + 1*7) / 5 = 3.4, seg1 = (1*7 + 2*10 + 2*13) / 5 = 10.6
    expected = np.array([[3.4, 4.4, 5.4], [10.6, 11.6, 12.6]])
    np.testing.assert_array_almost_equal(paa.paa(series, 2, "energy"), expected, decimal=10)
    # 'energy' and 'repeat' share the same per-column PAA; only SAX downstream differs.
    np.testing.assert_array_almost_equal(paa.paa(series, 2, "repeat"), expected, decimal=10)

    # paa_size == series_len is the identity (each point is its own segment).
    identity = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    np.testing.assert_array_almost_equal(paa.paa(identity, 3, "energy"), identity, decimal=10)

    # Divisible multi-dim: 4 rows -> 2 segments, block means of 2.
    divisible = np.array([[1, 5], [3, 7], [9, 1], [11, 3]], dtype=float)
    np.testing.assert_array_almost_equal(
        paa.paa(divisible, 2, "repeat"), np.array([[2.0, 6.0], [10.0, 2.0]]), decimal=10
    )


def test_paa3():
    """An irregular points to points via PAA."""
    dat = np.array(
        [
            -1.289433,
            -0.9992189,
            -0.5253246,
            -0.06612478,
            -0.2791935,
            0.08816637,
            -0.06612478,
            0.595123,
            0.8926845,
            0.8228861,
            1.741286,
            1.770675,
            -0.2791935,
            -1.197593,
            -1.208614,
        ]
    )
    dat_paa = np.array(
        [
            -1.192695,
            -0.6832894,
            -0.1371477,
            -0.03428692,
            0.1542912,
            0.7934974,
            1.129019,
            1.760878,
            -0.5853268,
            -1.20494,
        ]
    )
    np.testing.assert_array_almost_equal(paa.paa(dat, 10), dat_paa, decimal=6)
