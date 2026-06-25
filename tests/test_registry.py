"""Testing the visit registry implementation."""

import random
import time

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


def test_drain_visits_every_index_exactly_once():
    """Draining the registry must surface every index once and only once.

    Guards the swap-pop rewrite (audit #15): the O(1)-removal list/dict backing
    must not lose, duplicate, or invent indices.
    """
    reg = VisitRegistry(500)
    seen = []
    while True:
        idx = reg.get_next_unvisited()
        if isinstance(idx, float) and np.isnan(idx):
            break
        seen.append(int(idx))
        reg.mark_visited(int(idx))

    assert len(seen) == 500
    assert sorted(seen) == list(range(500))
    assert reg.get_unvisited_count() == 0


def test_mark_visited_is_idempotent_and_handles_absent():
    """Re-marking a visited (or never-present) index is a no-op, not a crash."""
    reg = VisitRegistry(10)
    reg.mark_visited(5)
    assert reg.get_unvisited_count() == 9
    reg.mark_visited(5)  # already gone
    assert reg.get_unvisited_count() == 9
    reg.mark_visited(999)  # never present
    assert reg.get_unvisited_count() == 9


def test_clone_is_independent():
    """A clone shares no mutable state with its source."""
    reg = VisitRegistry(10)
    clone = reg.clone()
    clone.mark_visited(5)
    assert reg.get_unvisited_count() == 10
    assert clone.get_unvisited_count() == 9


def test_get_next_unvisited_only_returns_unvisited():
    """Random selection must never return an already-visited index."""
    random.seed(0)
    reg = VisitRegistry(200)
    reg.mark_visited_range(0, 150)
    remaining = set(range(150, 200))
    for _ in range(500):
        idx = reg.get_next_unvisited()
        assert int(idx) in remaining


def test_drain_is_not_quadratic():
    """Perf guard (audit #15): the old set-based get_next_unvisited rebuilt
    ``tuple(self.remaining)`` on every call, making a single drain O(n^2) -- a
    50k drain took multiple seconds. The list/dict rewrite drains in O(n).

    The bound is deliberately loose (1 s vs the ~16 ms observed) so it only
    trips on a genuine return to quadratic behaviour, not on CI jitter.
    """
    reg = VisitRegistry(50000)
    start = time.perf_counter()
    while True:
        idx = reg.get_next_unvisited()
        if isinstance(idx, float) and np.isnan(idx):
            break
        reg.mark_visited(int(idx))
    assert time.perf_counter() - start < 1.0
