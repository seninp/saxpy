"""Discord discovery routines."""
import numpy as np
from saxpy.visit_registry import VisitRegistry
from saxpy.distance import early_abandoned_dist


def find_best_discord_brute_force(series, win_size):
    """Early-abandoned distance-based discord discovery."""
    best_so_far_distance = -1.0
    best_so_far_index = -1

    outerRegistry = VisitRegistry(len(series) - win_size)

    outer_idx = outerRegistry.get_next_unvisited()

    while ~np.isnan(outer_idx):

        outerRegistry.mark_visited(outer_idx)

        candidate_seq = series[outer_idx:(outer_idx+win_size)]

        nnDistance = np.inf
        innerRegistry = VisitRegistry(len(series) - win_size)

        inner_idx = innerRegistry.get_next_unvisited()

        while ~np.isnan(inner_idx):
            innerRegistry.mark_visited(inner_idx)

            if abs(inner_idx - outer_idx) > win_size:

                curr_seq = series[inner_idx:(inner_idx+win_size)]
                dist = early_abandoned_dist(candidate_seq,
                                            curr_seq, nnDistance)

                if ~np.isnan(dist) and (dist < nnDistance):
                    nnDistance = dist

            inner_idx = innerRegistry.get_next_unvisited()

        if ~(np.inf == nnDistance) and nnDistance > best_so_far_distance:
            best_so_far_distance = nnDistance
            best_so_far_index = outer_idx

        outer_idx = outerRegistry.get_next_unvisited()

    return (best_so_far_index, best_so_far_distance)


def find_best_discord_brute_force2(series, win_size):
    """Early-abandoned distance-based discord discovery."""
    best_so_far_distance = -1.0
    best_so_far_index = -1

    outerRegistry = VisitRegistry(len(series) - win_size)

    outer_idx = outerRegistry.get_next_unvisited2()

    while ~np.isnan(outer_idx):

        print("outer unvisited: " + str(outerRegistry.get_unvisited_count()))
        outerRegistry.mark_visited(outer_idx)

        candidate_seq = series[outer_idx:(outer_idx+win_size)]

        nnDistance = np.inf
        innerRegistry = VisitRegistry(len(series) - win_size)

        inner_idx = innerRegistry.get_next_unvisited2()

        while ~np.isnan(inner_idx):

            innerRegistry.mark_visited(inner_idx)

            if abs(inner_idx - outer_idx) > win_size:

                curr_seq = series[inner_idx:(inner_idx+win_size)]
                dist = early_abandoned_dist(candidate_seq,
                                            curr_seq, nnDistance)

                if ~np.isnan(dist) and (dist < nnDistance):
                    nnDistance = dist

            inner_idx = innerRegistry.get_next_unvisited2()

        if ~(np.inf == nnDistance) and nnDistance > best_so_far_distance:
            best_so_far_distance = nnDistance
            best_so_far_index = outer_idx

        outer_idx = outerRegistry.get_next_unvisited2()

    return (best_so_far_index, best_so_far_distance)
