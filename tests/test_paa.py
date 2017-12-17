"""Testing PAA implementation."""
import numpy as np
from saxpy import paa


def test_paa1():
    """Test points to points via PAA."""
    y = np.array([-1, -2, -1, 0, 2, 1, 1, 0])
    v = np.array([-1.375, 0.75, 0.625])
    np.testing.assert_array_equal(paa.paa(y, 3), v)
    np.testing.assert_array_equal(paa.paa(y, 8), y)
    np.testing.assert_array_equal(paa.paa(y, 2), np.array([-1, 1]))


def test_paa2():
    """An irregular points to points via PAA (15 -> 10)."""
    dat = np.array([-0.9796808, -0.8622706, -0.6123005, 0.8496459, 1.739691,
                    1.588194, 1.095829, 0.5277147, 0.4709033, -0.2865819,
                    0.0921607, -0.2865819, -0.9039323, -1.195564, -1.237226])
    dat_paa = np.array([-0.9405441, -0.6956239, 1.146328, 1.638693, 0.9064573,
                        0.4898404, -0.1603344, -0.1603344, -1.001143,
                        -1.223339])
    np.testing.assert_array_almost_equal(paa.paa(dat, 10),
                                         dat_paa, decimal=6)


def test_paa3():
    """An irregular points to points via PAA."""
    dat = np.array([-1.289433, -0.9992189, -0.5253246, -0.06612478, -0.2791935,
                    0.08816637, -0.06612478, 0.595123, 0.8926845, 0.8228861,
                    1.741286, 1.770675, -0.2791935, -1.197593, -1.208614])
    dat_paa = np.array([-1.192695, -0.6832894, -0.1371477, -0.03428692,
                        0.1542912, 0.7934974, 1.129019, 1.760878, -0.5853268,
                        -1.20494])
    np.testing.assert_array_almost_equal(paa.paa(dat, 10),
                                         dat_paa, decimal=6)
