"""Distance computation."""
import numpy as np


def euclidean(a, b):
    """Compute a Euclidean distance value."""
    return np.sqrt(np.sum((a-b)**2))
