"""Implements znorm."""

from __future__ import division
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import inv


def l2norm(array):
    """
    :param array: numpy array
    :return: non-negative real indicating the L2-norm of the array.

    >>> '%0.2f' % l2norm(np.array([1, 2, 3]))
    '3.74'
    >>> '%0.2f' % l2norm(np.array([1, 2, 3])[1])
    '2.00'
    >>> '%0.2f' % l2norm(np.array([1, 2, 3])[1:])
    '3.61'
    """

    return np.sqrt(np.sum(np.square(array)))


def znorm(series, znorm_threshold=0.01):
    """Znorm implementation.

    >>> print ['{:0.2f}'.format(x) for x in znorm([1, 2, 3])]
    ['-1.22', '0.00', '1.22']
    >>> print ['{:0.2f}'.format(x) for x in znorm([3, 2, 1])]
    ['1.22', '0.00', '-1.22']
    >>> print ['{:0.2f}'.format(x) for x in znorm([1, 2])]
    ['-1.00', '1.00']
    >>> print ['{:0.2f}'.format(x) for x in np.sum(znorm([[1, 2, 3], [6, 5, 4]]), axis=0)]
    ['0.00', '0.00', '0.00']
    >>> znorm([[1, 2, 3], [6, 5, 4]])
    array([[-1., -1., -1.],
           [ 1.,  1.,  1.]])
    """

    series = np.array(series)
    original_series_shape = series.shape
    is_multidimensional = (len(series.shape) > 1) and (series.shape[1] > 1)
    mu = np.average(series, axis=0)
    C = np.cov(series, bias=True, rowvar=not is_multidimensional)

    # Only update those subsequences with variance over the threshold.
    if is_multidimensional:
        series = series - mu
        C = np.diagonal(C)
        indexes = (C >= np.square(znorm_threshold))
        series[:, indexes] = (series[:, indexes] / np.sqrt(C[indexes]))
    else:
        series = series - mu
        if C >= np.square(znorm_threshold):
            series /= np.sqrt(C)

    # Check on shape returned.
    assert(series.shape == original_series_shape)

    return series
