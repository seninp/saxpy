"""Implements PAA."""
import numpy as np


def paa(series, paa_segments):
    """PAA implementation."""
    series_len = len(series)

    # check for the trivial case
    if (series_len == paa_segments):
        return np.copy(series)
    else:
        res = np.zeros(paa_segments)
        # check when we are even
        if (series_len % paa_segments == 0):
            inc = series_len // paa_segments
            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res, idx, series[i])
                # res[idx] = res[idx] + series[i]
            return res / inc
        # and process when we are odd
        else:
            for i in range(0, paa_segments * series_len):
                idx = i // series_len
                pos = i // paa_segments
                np.add.at(res, idx, series[pos])
                # res[idx] = res[idx] + series[pos]
            return res / series_len
