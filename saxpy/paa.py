"""Implements PAA."""

import math

import numpy as np


def _paa2(ts, paa_num):
    """Fractional-boundary PAA (matches jmotif-R ``_paa2`` / Java ``TSProcessor.paa``).

    Uses left-to-right sequential summation so segment means match the C++/Java
    reference even when a segment mean is ~0 and IEEE-754 order would flip its sign
    under NumPy's pairwise ``mean``.
    """
    ts = [float(x) for x in np.asarray(ts, dtype=float).ravel()]
    length = len(ts)
    if length == paa_num:
        return np.array(ts, dtype=float)

    points_per_segment = length / paa_num
    breaks = [i * points_per_segment for i in range(paa_num + 1)]
    breaks[paa_num] = float(length)

    res = []
    for i in range(paa_num):
        seg_start = breaks[i]
        seg_end = breaks[i + 1]

        frac_begin = math.ceil(seg_start) - seg_start
        frac_end = seg_end - math.floor(seg_end)

        full_begin = int(math.floor(seg_start))
        full_end = int(math.ceil(seg_end))
        if full_end > length:
            full_end = length

        sum_of_elems = 0.0
        for j in range(full_begin, full_end):
            v = ts[j]
            if j == full_begin and frac_begin > 0:
                v *= frac_begin
            if j == full_end - 1 and frac_end > 0:
                v *= frac_end
            sum_of_elems += v

        res.append(sum_of_elems / points_per_segment)

    return np.array(res, dtype=float)


def paa(series, paa_segment_size, sax_type="unidim"):
    """PAA implementation.

    >>> paa([1, 2, 3], 3, 'unidim')
    array([1., 2., 3.])
    >>> paa([1, 2, 3], 1, 'unidim')
    array([2.])
    >>> paa([4, 3, 8, 5], 1, 'unidim')
    array([5.])
    >>> paa([[1, 2, 3], [6, 5, 4]], 1, 'repeat')
    array([[3.5, 3.5, 3.5]])
    >>> paa([[1, 2, 3], [6, 5, 4]], 2, 'repeat')
    array([[1., 2., 3.],
           [6., 5., 4.]])
    """

    series = np.array(series)
    series_len = series.shape[0]

    # PAA reduces a series to fewer segments by averaging; reject inputs that
    # make that ill-defined instead of failing cryptically (ZeroDivisionError /
    # all-NaN) or silently up-sampling.
    if paa_segment_size < 1:
        raise ValueError("PAA segment size must be a positive integer.")
    if series_len == 0:
        raise ValueError("Cannot run PAA on an empty series.")
    if paa_segment_size > series_len:
        raise ValueError(
            "PAA segment size cannot exceed the series length; "
            "PAA reduces a series, it does not up-sample it."
        )

    if sax_type in ["repeat", "energy"]:
        num_dims = series.shape[1]
    else:
        num_dims = 1
        is_multidimensional = (len(series.shape) > 1) and (series.shape[1] > 1)
        if is_multidimensional:
            # A 1-D sax_type collapses to a single column, so a genuinely
            # multi-column array would silently drop every column but the first.
            # Reject it instead -- multi-dimensional input belongs to the
            # 'repeat', 'energy', or 'independent' modes.
            raise ValueError(
                f"sax_type={sax_type!r} expects a 1-D series, but got a "
                f"{series.shape[1]}-column array; use 'repeat', 'energy', or "
                "'independent' for multi-dimensional input."
            )
        series = series.reshape(series.shape[0], 1)

    res = np.zeros((num_dims, paa_segment_size))

    for dim in range(num_dims):
        res[dim] = _paa2(series[:, dim], paa_segment_size)

    if sax_type in ["repeat", "energy"]:
        return res.T
    else:
        return res.flatten()
