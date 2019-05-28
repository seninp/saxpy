"""Implements PAA."""
from __future__ import division
import numpy as np
from znorm import l2norm


def paa(series, paa_segment_size):
    """PAA implementation."""
    series = np.array(series)
    series_len = series.shape[0]

    # check for the trivial case
    if series_len == paa_segment_size:
        return np.copy(series)
    else:
        res = np.zeros(paa_segment_size)
        # check when we are even
        if series_len % paa_segment_size == 0:
            inc = series_len // paa_segment_size
            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res, idx, l2norm(series[i]))
                # res[idx] = res[idx] + series[i]
            return res / inc
        # and process when we are odd
        else:
            for i in range(0, paa_segment_size * series_len):
                idx = i // series_len
                pos = i // paa_segment_size
                np.add.at(res, idx, l2norm(series[pos]))
                # res[idx] = res[idx] + series[pos]
            return res / series_len
