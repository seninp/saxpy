"""Implements PAA."""
from __future__ import division
import numpy as np
from znorm import l2norm


def paa(series, paa_segment_size, sax_type='unidim'):
    """PAA implementation."""
    series = np.array(series)
    series_len = series.shape[0]

    if sax_type == 'repeat':
        num_dims = series.shape[1]
    else:
        num_dims = 1

    res = np.zeros((num_dims, paa_segment_size))

    for dim in range(num_dims):
        # Check if we can evenly divide
        if series_len % paa_segment_size == 0:
            inc = series_len // paa_segment_size
            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res[dim], idx, np.mean(series[i]))
            res[dim] /= inc
        # Process otherwise
        else:
            for i in range(0, paa_segment_size * series_len):
                idx = i // series_len
                pos = i // paa_segment_size
                np.add.at(res, idx, np.mean(series[pos]))
            res[dim] /= series_len

    if sax_type == 'repeat':
        return res
    else:
        return res.flatten()
