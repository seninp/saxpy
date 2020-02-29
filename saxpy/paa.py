"""Implements PAA."""
from __future__ import division
import numpy as np


def paa(series, paa_segment_size, sax_type='unidim'):
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

    if sax_type in ['repeat', 'energy']:
        num_dims = series.shape[1]
    else:
        num_dims = 1
        is_multidimensional = (len(series.shape) > 1) and (series.shape[1] > 1)
        if not is_multidimensional:
            series = series.reshape(series.shape[0], 1)

    res = np.zeros((num_dims, paa_segment_size))

    for dim in range(num_dims):
        # Check if we can evenly divide the series.
        if series_len % paa_segment_size == 0:
            inc = series_len // paa_segment_size

            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res[dim], idx, np.mean(series[i][dim]))
            res[dim] /= inc
        # Process otherwise.
        else:
            for i in range(0, paa_segment_size * series_len):
                idx = i // series_len
                pos = i // paa_segment_size
                np.add.at(res[dim], idx, np.mean(series[pos][dim]))
            res[dim] /= series_len

    if sax_type in ['repeat', 'energy']:
        return res.T
    else:
        return res.flatten()
