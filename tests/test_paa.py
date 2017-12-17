"""Testing PAA implementation."""
import numpy as np
from saxpy import paa


def test_znorm():
    """Test points to points via PAA"""
    y = np.array([-1, -2, -1, 0, 2, 1, 1, 0])
    v = np.array([-1.375, 0.75, 0.625])
    np.testing.assert_array_equal(paa.paa(y, 3), v)
    np.testing.assert_array_equal(paa.paa(y, 8), y)
    np.testing.assert_array_equal(paa.paa(y, 2), np.array([-1, 1]))
