"""Implements HOT-SAX."""
import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import sax_via_window
from saxpy.distance import euclidean


def find_discords_hotsax(series, win_size=100, num_discords=2, a_size=3,
                         paa_size=3, z_threshold=0.01):
    """HOT-SAX-driven discords discovery."""
    discords = list()

    globalRegistry = set()

    while (len(discords) < num_discords):

        bestDiscord = find_best_discord_hotsax(series, win_size, a_size,
                                               paa_size, z_threshold,
                                               globalRegistry)

        if -1 == bestDiscord[0]:
            break

        discords.append(bestDiscord)

        mark_start = bestDiscord[0] - win_size
        if 0 > mark_start:
            mark_start = 0

        mark_end = bestDiscord[0] + win_size
        '''if len(series) < mark_end:
            mark_end = len(series)'''

        for i in range(mark_start, mark_end):
            globalRegistry.add(i)

    return discords


def find_best_discord_hotsax(series, win_size, a_size, paa_size,
                             znorm_threshold, globalRegistry): # noqa: C901
    """Find the best discord with hotsax."""
    """[1.0] get the sax data first"""
    sax_none = sax_via_window(series, win_size, a_size, paa_size, "none", 0.01)

    """[2.0] build the 'magic' array"""
    magic_array = list()
    for k, v in sax_none.items():
        magic_array.append((k, len(v)))

    """[2.1] sort it desc by the key"""
    m_arr = sorted(magic_array, key=lambda tup: tup[1])

    """[3.0] define the key vars"""
    bestSoFarPosition = -1
    bestSoFarDistance = 0.

    distanceCalls = 0

    visit_array = np.zeros(len(series), dtype=np.int)

    """[4.0] and we are off iterating over the magic array entries"""
    for entry in m_arr:

        """[5.0] some moar of teh vars"""
        curr_word = entry[0]
        occurrences = sax_none[curr_word]

        """[6.0] jumping around by the same word occurrences makes it easier to
        nail down the possibly small distance value -- so we can be efficient
        and all that..."""
        for curr_pos in occurrences:

            if curr_pos in globalRegistry:
                continue

            """[7.0] we don't want an overlapping subsequence"""
            mark_start = curr_pos - win_size
            mark_end = curr_pos + win_size
            visit_set = set(range(mark_start, mark_end))

            """[8.0] here is our subsequence in question"""
            cur_seq = znorm(series[curr_pos:(curr_pos + win_size)],
                            znorm_threshold)

            """[9.0] let's see what is NN distance"""
            nn_dist = np.inf
            do_random_search = 1

            """[10.0] ordered by occurrences search first"""
            for next_pos in occurrences:

                """[11.0] skip bad pos"""
                if next_pos in visit_set:
                    continue
                else:
                    visit_set.add(next_pos)

                """[12.0] distance we compute"""
                dist = euclidean(cur_seq, znorm(series[next_pos:(
                                 next_pos+win_size)], znorm_threshold))
                distanceCalls += 1

                """[13.0] keep the books up-to-date"""
                if dist < nn_dist:
                    nn_dist = dist
                if dist < bestSoFarDistance:
                    do_random_search = 0
                    break

            """[13.0] if not broken above,
            we shall proceed with random search"""
            if do_random_search:
                """[14.0] build that random visit order array"""
                curr_idx = 0
                for i in range(0, (len(series) - win_size)):
                    if not(i in visit_set):
                        visit_array[curr_idx] = i
                        curr_idx += 1
                it_order = np.random.permutation(visit_array[0:curr_idx])
                curr_idx -= 1

                """[15.0] and go random"""
                while curr_idx >= 0:
                    rand_pos = it_order[curr_idx]
                    curr_idx -= 1

                    dist = euclidean(cur_seq, znorm(series[rand_pos:(
                                     rand_pos + win_size)], znorm_threshold))
                    distanceCalls += 1

                    """[16.0] keep the books up-to-date again"""
                    if dist < nn_dist:
                        nn_dist = dist
                    if dist < bestSoFarDistance:
                        nn_dist = dist
                        break

            """[17.0] and BIGGER books"""
            if (nn_dist > bestSoFarDistance) and (nn_dist < np.inf):
                bestSoFarDistance = nn_dist
                bestSoFarPosition = curr_pos

    return (bestSoFarPosition, bestSoFarDistance)
