import numpy as np
import pytest
import saxpy as sp

def test_znorm():

    ts = [-1., -2., -1., 0., 2., 1., 1., 0.]
    z_thrsh = 0.001

    x_scaled = map(lambda i: i/100.0, ts)

    assert pytest.approx(1.0, 0.000001) == \
                        np.std(sp.znorm(x_scaled, z_thrsh), axis=0, ddof=1)
