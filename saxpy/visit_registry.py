"""Keeps visited indexes in check."""
import numpy as np
import random


class VisitRegistry:
    """A straightforward visit array implementation."""

    def __init__(self, capacity=0):
        """Constructor."""
        self.remaining = set()
        for num in range(capacity):
            self.remaining.add(num)

    def get_unvisited_count(self):
        """An unvisited count getter."""
        return len(self.remaining)

    def mark_visited(self, index):
        """Set a single index as visited."""
        self.remaining.discard(index)

    def mark_visited_range(self, start, stop):
        """Set a range as visited."""
        for i in range(start, stop):
            self.mark_visited(i)

    def get_next_unvisited(self):
        """Next unvisited entry."""
        if self.get_unvisited_count() == 0:
            return np.nan

        return random.choice(tuple(self.remaining))

    def clone(self):
        """Make the array's copy."""
        clone = VisitRegistry()
        clone.remaining = self.remaining.copy()
        return clone
