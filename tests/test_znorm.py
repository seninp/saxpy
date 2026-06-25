"""Testing znorm implementation."""

import numpy as np
import pytest
from numpy import array, mean, std

from saxpy import znorm


def test_znorm():
    """Test the znorm implementation."""
    # test std is 1 and mean is 0
    ts = array([-1.0, -2.0, -1.0, 0.0, 2.0, 1.0, 1.0, 0.0])
    z_thrsh = 0.001
    x_scaled = [x / 100.0 for x in ts]
    assert pytest.approx(1.0, 0.000001) == std(znorm.znorm(x_scaled, z_thrsh))
    assert pytest.approx(0.0, 0.000001) == mean(znorm.znorm(x_scaled, z_thrsh))

    # test std and mean wouldnt change on hi threshold
    ts = array([-0.1, -0.2, 0.2, 0.1])
    z_thrsh = 0.5
    ts_mean = mean(ts)
    ts_sd = std(ts)
    assert ts_mean == mean(znorm.znorm(ts, z_thrsh))
    assert ts_sd == std(znorm.znorm(ts, z_thrsh))


def test_znorm_column_vector():
    """Regression (audit #5): a 2-D univariate (n, 1) column vector must
    z-normalize, not crash. The old np.cov(..., rowvar=True) on (n, 1) returned
    an (n, n) matrix, making the scalar threshold test raise "truth value of an
    array ... is ambiguous". A column vector must give the same result as its
    1-D form.
    """
    col = znorm.znorm(array([[1.0], [2.0], [3.0]]))
    assert col.shape == (3, 1)
    assert np.allclose(col.ravel(), znorm.znorm(array([1.0, 2.0, 3.0])))
