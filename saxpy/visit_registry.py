"""Keeps visited indexes in check."""

import random

import numpy as np


class VisitRegistry:
    """Tracks unvisited indices with O(1) random selection and removal.

    Backed by a dense list of the currently-unvisited indices plus a
    value -> position map, so removal is O(1) (swap-with-last then pop) and a
    random pick is O(1) (a list is directly indexable). The previous set-based
    implementation rebuilt ``tuple(self.remaining)`` on every
    ``get_next_unvisited`` call, turning a single drain into O(n^2) work (and
    O(n^3) in the brute-force discord search, which drains a fresh inner
    registry per outer index).
    """

    def __init__(self, capacity=0, rng=None):
        """Constructor.

        ``rng`` is an optional ``random.Random`` instance used by
        :meth:`get_next_unvisited`. When ``None`` the module-global ``random``
        is used (the historical, unseeded behavior). Pass a seeded
        ``random.Random`` to make the visit order -- and hence the
        early-abandoning search trajectory and its distance-call count --
        reproducible. The discord *result* is order-independent either way.
        """
        self._items = list(range(capacity))
        self._pos = {value: i for i, value in enumerate(self._items)}
        self._rng = rng

    def get_unvisited_count(self):
        """An unvisited count getter."""
        return len(self._items)

    def mark_visited(self, index):
        """Set a single index as visited."""
        i = self._pos.pop(index, None)
        if i is None:
            return
        last = self._items.pop()
        if i != len(self._items):
            self._items[i] = last
            self._pos[last] = i

    def mark_visited_range(self, start, stop):
        """Set a range as visited."""
        for i in range(start, stop):
            self.mark_visited(i)

    def get_next_unvisited(self):
        """A randomly chosen unvisited entry, or ``np.nan`` when none remain."""
        if not self._items:
            return np.nan

        chooser = self._rng if self._rng is not None else random
        return chooser.choice(self._items)

    def clone(self):
        """Make the registry's copy (sharing the same ``rng`` instance)."""
        clone = VisitRegistry()
        clone._items = self._items.copy()
        clone._pos = self._pos.copy()
        clone._rng = self._rng
        return clone
