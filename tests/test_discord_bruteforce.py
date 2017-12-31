"""Testing brute force discord search implementation."""
import numpy as np
from saxpy.discord import find_discords_brute_force


def test_brute_force():
    """Test brute force discord discovery."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=',')
    discords = find_discords_brute_force(dd[0:400], 100, 4)
    assert 3 == len(discords)
    assert 173 == discords[0][0]
