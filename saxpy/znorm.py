"""Implements znorm."""
import numpy as np


def znorm(series, znorm_threshold=0.01):
    """Znorm implementation."""
    sd = np.std(series)
    if (sd < znorm_threshold):
        return series
    mean = np.mean(series)
    return (series - mean) / sd


def znorms(full_series, win_size, znorm_threshold=0.01):
    idx = np.arange(win_size)[None, :] \
            + np.arange(len(full_series) - win_size + 1)[:, None]
    stds = np.expand_dims(np.std(full_series[idx], axis=1), axis=1)
    stds[stds < znorm_threshold] = 1.0
    means = np.expand_dims(np.mean(full_series[idx], axis=1), axis=1)
    means[stds < znorm_threshold] = 0.0

    znorms = (full_series[idx] - means) / stds
    return znorms
