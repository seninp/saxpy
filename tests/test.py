import pytest
import numpy as np
import saxpy as sp

def test_znorm():

    ts = [-1., -2., -1., 0., 2., 1., 1., 0.]
    z_thrsh = 0.001

    x_scaled =[x/100.0 for x in ts]

    assert pytest.approx(1.0, 0.000001) == \
                        np.std(sp.znorm(x_scaled, z_thrsh), axis=0, ddof=1)
