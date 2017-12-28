"""Testing HOTSAX discord search implementation."""
import numpy as np
from saxpy.hotsax import find_best_discord_hotsax


def test_brute_force():
    """Test brute force discord discovery."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=',')
    discord = find_best_discord_hotsax(dd, 100)
    assert 430 == discord[0]
