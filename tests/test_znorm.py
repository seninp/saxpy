"""Testing znorm implementation."""
import pytest
from numpy import array, std, mean
from saxpy import znorm


def test_znorm():
    """Test the znorm implementation."""
    # test std is 1 and mean is 0
    ts = array([-1., -2., -1., 0., 2., 1., 1., 0.])
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
