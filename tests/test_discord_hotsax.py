"""Testing HOTSAX discord search implementation."""
import numpy as np
from saxpy.hotsax import find_discords_hotsax


def test_brute_force():
    """Test HOT-SAX discord discovery."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=',')

    discords = find_discords_hotsax(dd)
    assert 430 == discords[0][0]
    assert 318 == discords[1][0]

    discords = find_discords_hotsax(dd[0:400], 100, 4)
    assert 3 == len(discords)
    assert 173 == discords[0][0]
