"""Implements znorm."""

import numpy as np


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
    mu = np.average(series, axis=0)

    # Only update those subsequences with variance over the threshold.
    if is_multidimensional:
        C = np.diagonal(np.cov(series, bias=True, rowvar=False))
        series = series - mu
        indexes = C >= np.square(znorm_threshold)
        series[:, indexes] = series[:, indexes] / np.sqrt(C[indexes])
    else:
        # Population variance over the sample axis. Using np.var rather than
        # np.cov(..., rowvar=True) keeps a 2-D column vector (n, 1) univariate:
        # np.cov on an (n, 1) array returns an (n, n) matrix, which made the
        # scalar threshold test raise "truth value ambiguous".
        C = np.var(series, axis=0)
        series = series - mu
        if np.all(C >= np.square(znorm_threshold)):
            series = series / np.sqrt(C)

    # Check on shape returned.
    assert series.shape == original_series_shape

    return series
