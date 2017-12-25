"""Testing PAA implementation."""
import pytest
import numpy as np
from saxpy import distance


def test_euclidean():
    """Test euclidean distance."""
    assert pytest.approx(np.sqrt(2), 0.0000001) == distance.euclidean(
                         np.array([1., 1.]), np.array([2., 2.]))

    a = np.array([0.5, 0.8, 0.9])
    b = np.array([-0.15, 0.38, 0.92])
    assert pytest.approx(0.7741447, 0.0000001) == distance.euclidean(a, b)


def test_early_abandoned():
    """Test euclidean distance."""
    assert pytest.approx(np.sqrt(2), 0.0000001) ==\
        distance.early_abandoned_dist(np.array([1., 1.]),
                                      np.array([2., 2.]), np.inf)

    a = np.array([0.5, 0.8, 0.9])
    b = np.array([-0.15, 0.38, 0.92])
    assert pytest.approx(0.7741447, 0.0000001) ==\
        distance.early_abandoned_dist(a, b, np.inf)

    assert 1 == np.isnan(distance.early_abandoned_dist(a, b, 0.1))
