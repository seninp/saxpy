"""Implements Alphabet cuts."""

from functools import cache

import numpy as np
from scipy.stats import norm  # already depend on scipy


@cache
def cuts_for_asize(a_size: int) -> np.ndarray:
    """Generate the Gaussian breakpoints for an alphabet of the given size.

    The returned array is prefixed with ``-inf`` and has ``a_size`` entries.

    >>> cuts_for_asize(2)
    array([-inf,   0.])
    >>> cuts_for_asize(3)
    array([      -inf, -0.4307273,  0.4307273])
    >>> len(cuts_for_asize(5))
    5
    """
    if a_size < 2:
        raise ValueError("alphabet_size must be >= 2")
    probs = np.arange(1, a_size) / a_size
    cuts = norm.ppf(probs)
    return np.concatenate(([-np.inf], cuts))
