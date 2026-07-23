"""Tests for bundled sample data."""

import numpy as np

from saxpy.datasets import ecg0606_path


def test_ecg0606_path_is_readable():
    path = ecg0606_path()
    assert path.is_file()
    series = np.genfromtxt(path, delimiter=",")
    assert series.shape == (2299,)
