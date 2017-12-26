"""Testing PAA implementation."""
import pytest
import numpy as np
from saxpy.discord import find_best_discord_brute_force


def test_brute_force():
    """Test brute force discord discovery."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=',')
    discord = find_best_discord_brute_force(dd[0:400], 100)
    assert 173 == discord[0]
