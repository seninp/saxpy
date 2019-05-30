"""Distance computation."""
import numpy as np


def euclidean(a, b):
    """Compute a Euclidean distance value."""
    return np.sqrt(np.sum(np.square(a-b)))


def early_abandoned_euclidean(a, b, upper_limit):
    """Compute a Euclidean distance value in early abandoning fashion."""
    lim = np.square(upper_limit)
    res = 0.
    for i in range(0, len(a)):
        res += np.sum(np.square((a[i]-b[i])))
        # if res > lim:
        #     return np.nan
    return np.sqrt(res)
