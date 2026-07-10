"""Implements znorm."""

import numpy as np


def _znorm_univariate(series, znorm_threshold):
    """Z-normalize a 1-D series with C++/R-style sequential accumulation.

    NumPy's ``mean`` / ``var`` use pairwise summation; on z-normalized windows whose
    last PAA segment mean is ~0 that can flip the sign at machine epsilon and change
    the SAX symbol (see ecg0606 index 87, w=100, paa=4, a=4).
    """
    data = [float(x) for x in np.asarray(series, dtype=float).ravel()]
    n = len(data)
    if n == 0:
        raise ValueError("Cannot z-normalize an empty series.")

    total = 0.0
    for x in data:
        total += x
    mu = total / n

    sq_sum = 0.0
    for x in data:
        d = x - mu
        sq_sum += d * d
    var = sq_sum / n

    centered = [x - mu for x in data]
    if var >= znorm_threshold**2:
        inv_std = 1.0 / (var**0.5)
        return np.array([c * inv_std for c in centered], dtype=float)
    return np.array(centered, dtype=float)


def znorm(series, znorm_threshold=0.01):
    """Znorm implementation.

    >>> print(['{:0.2f}'.format(x) for x in znorm([1, 2, 3])])
    ['-1.22', '0.00', '1.22']
    >>> print(['{:0.2f}'.format(x) for x in znorm([3, 2, 1])])
    ['1.22', '0.00', '-1.22']
    >>> print(['{:0.2f}'.format(x) for x in znorm([1, 2])])
    ['-1.00', '1.00']
    >>> print(['{:0.2f}'.format(x) for x in np.sum(znorm([[1, 2, 3], [6, 5, 4]]), axis=0)])
    ['0.00', '0.00', '0.00']
    >>> znorm([[1, 2, 3], [6, 5, 4]])
    array([[-1., -1., -1.],
           [ 1.,  1.,  1.]])
    """

    series = np.array(series)
    original_series_shape = series.shape
    is_multidimensional = (len(series.shape) > 1) and (series.shape[1] > 1)
    if not is_multidimensional:
        out = _znorm_univariate(series, znorm_threshold)
        if len(original_series_shape) > 1 and original_series_shape[1] == 1:
            out = out.reshape(-1, 1)
        assert out.shape == original_series_shape
        return out

    mu = np.average(series, axis=0)

    # Only update those subsequences with variance over the threshold.
    C = np.diagonal(np.cov(series, bias=True, rowvar=False))
    series = series - mu
    indexes = C >= np.square(znorm_threshold)
    series[:, indexes] = series[:, indexes] / np.sqrt(C[indexes])

    # Check on shape returned.
    assert series.shape == original_series_shape

    return series
