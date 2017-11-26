import pytest
import numpy as np

def test_znorm():

    ts = [-1., -2., -1., 0., 2., 1., 1., 0.]
    z_thrsh = 0.001

    x_scaled =[x/100.0 for x in ts]

    import saxpy as sp

    assert pytest.approx(1.0, 0.000001) == \
                        np.std(sp.znorm(x_scaled, z_thrsh), axis=0, ddof=1)

def capital_case(x):
    return x.capitalize()

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'
