"""Implements znorm."""
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

    >>> print ['{:0.2f}'.format(x) for x in znorm(np.array([1, 2, 3]))]
    ['-1.22', '0.00', '1.22']
    >>> print ['{:0.2f}'.format(x) for x in znorm(np.array([3, 2, 1]))]
    ['1.22', '0.00', '-1.22']
    >>> print ['{:0.2f}'.format(x) for x in znorm(np.array([1, 2]))]
    ['-1.00', '1.00']
    >>> print ['{:0.2f}'.format(x) for x in np.sum(znorm(np.array([[1, 2, 3], [6, 5, 4]])), axis=0)]
    ['0.00', '0.00', '0.00']
    """

    series = np.array(series)
    multidim = (len(series.shape) > 1)
    mu = np.average(series, axis=0)
    C = np.cov(series, bias=True, rowvar=not multidim)

    if len(C.shape) == 0:
        C = np.array([[C]])

    if np.any(np.diagonal(C) < np.square(znorm_threshold)):
        return series

    if multidim:
        return np.dot(inv(sqrtm(np.matrix(C))), (series - mu).T).T
    else:
        return np.multiply(inv(sqrtm(np.matrix(C))), (series - mu)).flatten()


