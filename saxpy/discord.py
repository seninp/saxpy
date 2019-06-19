"""Discord discovery routines."""
import numpy as np
from saxpy.visit_registry import VisitRegistry
from saxpy.distance import early_abandoned_euclidean
from saxpy.znorm import znorm


def find_discords_brute_force(series, win_size, num_discords=2, znorm_threshold=0.01):
    """Early-abandoned distance-based discord discovery."""
    discords = list()

    globalRegistry = VisitRegistry(len(series) - win_size + 1)
    znorms = np.array([znorm(series[pos: pos + win_size], znorm_threshold) for pos in range(len(series) - win_size + 1)])

    while len(discords) < num_discords:

        bestDiscord = find_best_discord_brute_force(series, win_size,
                                                    globalRegistry,
                                                    znorms)

        if -1 == bestDiscord[0]:
            break

        discords.append(bestDiscord)

        mark_start = max(0, bestDiscord[0] - win_size + 1)
        mark_end = bestDiscord[0] + win_size

        globalRegistry.mark_visited_range(mark_start, mark_end)

    return discords


def find_best_discord_brute_force(series, win_size, global_registry, znorms):
    """Early-abandoned distance-based discord discovery."""
    best_so_far_distance = -1.0
    best_so_far_index = -1

    outerRegistry = global_registry.clone()

    outer_idx = outerRegistry.get_next_unvisited()

    while ~np.isnan(outer_idx):

        outerRegistry.mark_visited(outer_idx)

        candidate_seq = znorms[outer_idx]

        nnDistance = np.inf
        innerRegistry = VisitRegistry(len(series) - win_size + 1)

        inner_idx = innerRegistry.get_next_unvisited()

        while ~np.isnan(inner_idx):
            innerRegistry.mark_visited(inner_idx)

            if abs(inner_idx - outer_idx) >= win_size:

                curr_seq = znorms[inner_idx]

                dist = early_abandoned_euclidean(candidate_seq, curr_seq, nnDistance)

                if (~np.isnan(dist)) and (dist < nnDistance):
                    nnDistance = dist

            inner_idx = innerRegistry.get_next_unvisited()

        if ~(np.inf == nnDistance) and (nnDistance > best_so_far_distance):
            best_so_far_distance = nnDistance
            best_so_far_index = outer_idx

        outer_idx = outerRegistry.get_next_unvisited()

    return best_so_far_index, best_so_far_distance