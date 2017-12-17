"""Implements znorm."""
import numpy as np


def znorm(series, znorm_threshold=0.01):
    """Znorm implementation."""
    sd = np.std(series)
    if (sd < znorm_threshold):
        return series
    mean = np.mean(series)
    return (series - mean) / sd
