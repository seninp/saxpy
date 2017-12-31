"""Keeps visited indexes in check."""
import numpy as np


class VisitRegistry:
    """A straightforward visit array implementation."""

    def __init__(self, capacity):
        """Constructor."""
        self.capacity = capacity
        self.visit_array = np.zeros(capacity, dtype=np.uint8)
        self.unvisited_count = capacity

    def get_unvisited_count(self):
        """An unvisited count getter."""
        return self.unvisited_count

    def mark_visited(self, index):
        """Set a single index as visited."""
        if (0 == self.visit_array[index]):
            self.visit_array[index] = 1
            self.unvisited_count -= 1

    def mark_visited_range(self, start, stop):
        """Set a range as visited."""
        for i in range(start, stop):
            self.mark_visited(i)

    def get_next_unvisited(self):
        """Memory-optimized version."""
        if 0 == self.unvisited_count:
            return np.nan
        else:
            i = np.random.randint(0, self.capacity)
            while 1 == self.visit_array[i]:
                i = np.random.randint(0, self.capacity)
            return i

    def clone(self):
        """Make the array's copy."""
        clone = VisitRegistry(self.capacity)
        setattr(clone, 'visit_array', self.visit_array.copy())
        setattr(clone, 'unvisited_count', self.unvisited_count)
        return clone

    '''def get_next_unvisited2(self):
        """Speed-optimized version."""
        if 0 == self.unvisited_count:
            return np.nan
        else:
            tmp_order = np.zeros(self.unvisited_count, dtype=np.int32)
            j = 0
            for i in range(0, self.capacity):
                if 0 == self.visit_array[i]:
                    tmp_order[j] = i
                    j += 1
            np.random.shuffle(tmp_order)
            return tmp_order[0]'''
