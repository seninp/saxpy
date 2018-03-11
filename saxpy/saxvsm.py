"""Implements VSM."""
import numpy as np
from saxpy.sax import sax_via_window


def series_to_wordbag(series, win_size, paa_size, alphabet_size=3,
                      nr_strategy='exact', z_threshold=0.01):
    """VSM implementation."""
    sax = sax_via_window(series, win_size, paa_size, alphabet_size,
                         nr_strategy, z_threshold)

    # convert the dict to a wordbag
    frequencies = {}
    for k, v in sax.items():
        frequencies[k] = len(v)
    return frequencies


def manyseries_to_wordbag(series_npmatrix, win_size, paa_size, alphabet_size=3,
                          nr_strategy='exact', z_threshold=0.01):
    """VSM implementation."""
    frequencies = {}

    for row in series_npmatrix:
        tmp_freq = series_to_wordbag(np.squeeze(np.asarray(row)),
                                     win_size, paa_size, alphabet_size,
                                     nr_strategy, z_threshold)
        for k, v in tmp_freq.items():
            if k in frequencies:
                frequencies[k] += v
            else:
                frequencies[k] = v

    return frequencies
