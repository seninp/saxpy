"""Distance computation."""
import numpy as np


def euclidean(a, b):
    """Compute a Euclidean distance value."""
    return np.sqrt(np.sum((a-b)**2))


def early_abandoned_euclidean(a, b, upper_limit):
    """Compute a Euclidean distance value in early abandoning fashion."""
    lim = upper_limit * upper_limit
    res = 0.
    for i in range(0, len(a)):
        res += np.dot((a[i]-b[i]), (a[i]-b[i]))
        if res > lim:
            return np.nan
    return np.sqrt(res)
