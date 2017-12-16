"""Testing znorm implementation."""
import pytest
from numpy import std
from saxpy import znorm


def test_znorm():
    """Test the znorm implementation."""
    ts = [-1., -2., -1., 0., 2., 1., 1., 0.]
    z_thrsh = 0.001

    x_scaled = [x / 100.0 for x in ts]

    assert pytest.approx(1.0, 0.000001) == \
        std(znorm.znorm(x_scaled, z_thrsh), axis=0, ddof=1)

#    assert pytest.approx(1.0, 0.000001) == \
#                    np.std(sp.znorm_py(x_scaled, z_thrsh), axis=0, ddof=1)
