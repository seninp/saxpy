"""Testing PAA implementation."""
import numpy as np
from saxpy.visit_registry import VisitRegistry


def test_sizing():
    """Test the registry."""
    reg = VisitRegistry(77)
    assert 77 == reg.get_unvisited_count()

    reg.mark_visited(0)
    assert 76 == reg.get_unvisited_count()

    reg.mark_visited_range(70, 77)
    assert 69 == reg.get_unvisited_count()

    reg.mark_visited(0)
    assert 69 == reg.get_unvisited_count()
    reg.mark_visited(1)
    assert 68 == reg.get_unvisited_count()

    reg.mark_visited(reg.get_next_unvisited())
    assert 67 == reg.get_unvisited_count()

    reg.mark_visited_range(0, 77)
    assert np.isnan(reg.get_next_unvisited())
