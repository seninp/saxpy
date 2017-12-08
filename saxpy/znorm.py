"""Implements znorm."""
import numpy as np


def znorm(series, znorm_threshold=0.01):
    """Fallback Python implementation."""
    sd = np.std(series)
    if (sd < znorm_threshold):
        return series
    mean = np.mean(series)
    res = np.zeros(len(series))
    for i in range(0, len(series)):
        res[i] = (series[i] - mean) / sd
    return res


try:
    from ._saxpy import znorm as _znorm
    znorm = _znorm # NOQA
except ImportError:
    pass
