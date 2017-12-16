"""Implements znorm."""
import numpy as np


def znorm(series, znorm_threshold=0.01):
    """Fallback Python implementation."""
    sd = np.std(series)
    if (sd < znorm_threshold):
        return series
    mean = np.mean(series)
    return (series - mean) / sd


try:
    from ._saxpy import znorm as _znorm
    znorm = _znorm # NOQA
except ImportError:
    pass
