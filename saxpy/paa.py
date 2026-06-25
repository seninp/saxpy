"""Implements PAA."""

import numpy as np


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
        column = series[:, dim]
        # PAA by averaging. These are the vectorized form of the original
        # element-wise ``np.add.at`` scatter loops -- same arithmetic and
        # summation order, but orders of magnitude faster on long series.
        if series_len % paa_segment_size == 0:
            # Evenly divisible: average contiguous blocks of ``inc`` points.
            inc = series_len // paa_segment_size
            res[dim] = column.reshape(paa_segment_size, inc).mean(axis=1)
        else:
            # Otherwise the classic expand-by-paa_size / contract-by-series_len
            # construction, so segment boundaries can fall between samples.
            res[dim] = (
                np.repeat(column, paa_segment_size)
                .reshape(paa_segment_size, series_len)
                .mean(axis=1)
            )

    if sax_type in ["repeat", "energy"]:
        return res.T
    else:
        return res.flatten()
