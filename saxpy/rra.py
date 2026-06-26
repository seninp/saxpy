"""RRA (Rare Rule Anomaly) discord discovery.

A pure-Python port of the RRA algorithm from the sibling jMotif C++/R project
(Senin et al., *Time series anomaly discovery with grammar-based compression*,
EDBT 2015). RRA is HOT-SAX refined with grammatical compression: instead of
fixed-length sliding-window subsequences, it searches over the variable-length
subsequences that correspond to the rules of a RePair grammar
(:func:`saxpy.repair.str_to_repair_grammar`) built from the SAX representation.

The intuition: a grammar compresses recurring structure into rules, so the
*rule coverage* of each point reflects how compressible (i.e. how typical) it is.
Rare, low-coverage subsequences are the anomaly candidates, and RRA orders the
nearest-neighbour search by coverage (rare first) -- the grammar analogue of
HOT-SAX's word-rarity ordering, over far fewer candidates.

The entry point is :func:`find_discords_rra`. It is the variable-length
counterpart to :func:`saxpy.hotsax.find_discords_hotsax`.
"""

import random
from dataclasses import dataclass

import numpy as np

from saxpy.paa import paa as _paa
from saxpy.repair import str_to_repair_grammar
from saxpy.sax import sax_via_window


@dataclass
class RRADiscord:
    """A discovered RRA discord: a rule, its time-series span, and NN distance."""

    rule_id: int
    start: int
    end: int
    length: int
    nn_distance: float


def _paa2(ts, paa_num):
    """Fractional-boundary PAA used by RRA's length-normalizing distance.

    This is exactly :func:`saxpy.paa.paa` (overlap-weighted / fractional-boundary
    PAA): boundaries may fall between samples, and straddling samples are
    weighted by their fractional overlap. RRA uses it only to rescale the longer
    of two unequal-length subsequences down to the shorter length, so ``paa_num``
    is always ``<= len(ts)`` here.

    It delegates to ``saxpy.paa.paa`` -- the standalone loop this used to carry
    was a port of the C++ ``_paa2`` that (a) spent ~95% of ``find_discords_rra``'s
    runtime in tiny per-segment NumPy calls and (b) dropped the final sample when
    ``len/paa_num`` rounded just past an integer (the float-boundary bug the C++
    later fixed by snapping the last break to ``len``). ``saxpy.paa`` is the
    vectorized, correct fractional PAA and matches the C++ reference, so RRA now
    reuses it rather than maintaining a second copy.
    """
    ts = np.asarray(ts, dtype=float)
    if len(ts) < paa_num:
        raise ValueError("'paa_num' size is invalid")
    return _paa(ts, paa_num)


def _normalized_distance(start1, end1, start2, end2, series):
    """Per-point normalized Euclidean distance between two raw subsequences.

    Operates on the RAW series (not z-normed). When the two spans differ in
    length, the longer is PAA-reduced (via :func:`_paa2`) to the shorter length
    first. The result is ``sqrt(sum of squared diffs) / count`` -- divided by the
    number of compared points, so spans of different lengths are comparable.
    """
    len1 = end1 - start1
    len2 = end2 - start2

    if len1 == len2:
        a = series[start1:end1]
        b = series[start2:end2]
        diff = a - b
        return float(np.sqrt(np.dot(diff, diff)) / len1)

    min_length = min(len1, len2)
    if len1 == min_length:
        shorter = series[start1:end1]
        longer_paa = _paa2(series[start2:end2], len1)
    else:
        shorter = series[start2:end2]
        longer_paa = _paa2(series[start1:end1], len2)
    diff = shorter - longer_paa
    return float(np.sqrt(np.dot(diff, diff)) / min_length)


def _find_best_rra_discord(series, win_size, grammar, indexes, intervals, global_visited, rng):
    """Find the single farthest-nearest-neighbour rule interval (one discord)."""
    best_position = -1
    best_length = -1
    best_rule = 0
    best_distance = -1.0

    n_intervals = len(intervals)

    for c in intervals:
        if c.start in global_visited:
            continue

        # Exclusion zone scales with this interval's length (variable-length).
        visited = set()
        mark_start = max(0, c.start - (c.end - c.start))
        mark_end = min(len(series), c.end)
        visited.update(range(mark_start, mark_end))

        nn_distance = np.inf
        do_random_search = True

        # Search same-rule occurrences first (likely close), early-abandoning.
        for occ_start, occ_end in grammar[c.rule_id].intervals:
            start = indexes[occ_start]
            if start in visited:
                continue
            visited.add(start)
            end = indexes[occ_end] + win_size
            dist = _normalized_distance(c.start, c.end, start, end, series)
            if dist < nn_distance:
                nn_distance = dist
            if dist < best_distance:
                do_random_search = False
                break

        if do_random_search:
            # Visit the remaining intervals in random order (a speed heuristic;
            # the farthest-NN result is order-independent except on exact ties).
            candidates = [j for j in range(n_intervals) if intervals[j].start not in visited]
            rng.shuffle(candidates)
            for j in candidates:
                ri = intervals[j]
                dist = _normalized_distance(c.start, c.end, ri.start, ri.end, series)
                if dist < best_distance:
                    nn_distance = dist
                    break
                if dist < nn_distance:
                    nn_distance = dist

        if nn_distance > best_distance:
            best_distance = nn_distance
            best_position = c.start
            best_length = c.end - c.start
            best_rule = c.rule_id

    return RRADiscord(
        rule_id=best_rule,
        start=best_position,
        end=best_position + best_length,
        length=best_length,
        nn_distance=best_distance,
    )


@dataclass
class _Interval:
    rule_id: int
    start: int
    end: int
    cover: float = 0.0


def find_discords_rra(
    series,
    win_size,
    paa_size=3,
    alphabet_size=3,
    nr_strategy="none",
    znorm_threshold=0.01,
    num_discords=2,
):
    """Find discords (anomalies) with the Rare Rule Anomaly algorithm.

    Builds a RePair grammar over the SAX representation of ``series``, derives
    variable-length subsequences from the grammar rules, and searches them
    rarest-first for the subsequence whose nearest neighbour is farthest away.
    Tends to work best with larger PAA/alphabet sizes than HOT-SAX.

    Args:
        series: the 1-D input time series.
        win_size: the sliding window size for the SAX step.
        paa_size: the PAA size.
        alphabet_size: the SAX alphabet size.
        nr_strategy: numerosity-reduction strategy (``"none"``, ``"exact"``,
            ``"mindist"``); ``"none"`` (the default, matching the reference)
            keeps every window so positions stay contiguous.
        znorm_threshold: the z-normalization variance threshold.
        num_discords: how many discords to report.

    Returns:
        A list of :class:`RRADiscord`, ordered by discovery (most anomalous
        first). May be shorter than ``num_discords`` if candidates run out.

    The grammar's rule numbering and the random search order are not portable
    across implementations, but the reported discord regions are: see
    :func:`saxpy.repair.str_to_repair_grammar`.
    """
    series = np.asarray(series, dtype=float)
    if win_size < 1:
        raise ValueError("win_size must be a positive integer.")
    if len(series) < win_size:
        raise ValueError("win_size cannot exceed the series length.")

    # SAX words per window. Pass nr_strategy through explicitly -- sax_via_window
    # defaults to 'exact', which would collapse equal-word runs and break the
    # position->word string composition below.
    sax_map = sax_via_window(
        series,
        win_size,
        paa_size,
        alphabet_size=alphabet_size,
        nr_strategy=nr_strategy,
        znorm_threshold=znorm_threshold,
    )

    # Invert word->[positions] to position->word, then order by position.
    position_to_word = {}
    for word, positions in sax_map.items():
        for p in positions:
            position_to_word[p] = word
    indexes = sorted(position_to_word)
    if not indexes:
        return []

    sax_str = " ".join(position_to_word[p] for p in indexes)

    grammar = str_to_repair_grammar(sax_str)

    # Map each grammar rule's token intervals to time-series intervals, and
    # accumulate a per-point coverage curve.
    intervals = []
    coverage = np.zeros(len(series), dtype=int)
    for rid, rule in grammar.items():
        if rid == 0:
            continue  # skip R0
        for t_start, t_end in rule.intervals:
            start = indexes[t_start]
            end = indexes[t_end] + win_size
            coverage[start:end] += 1
            intervals.append(_Interval(rule_id=rid, start=start, end=end))

    # Uncovered (zero-coverage) stretches become their own rule_id == -1
    # candidate intervals -- these are the most anomalous by construction.
    zero_rule = grammar.get(-1)
    if zero_rule is None:
        from saxpy.repair import RepairRule

        zero_rule = RepairRule(rule_id=-1, rule_string="xxx", expanded_rule_string="xxx")
    in_interval = False
    start = -1
    for i in range(len(coverage)):
        if coverage[i] == 0 and not in_interval:
            start = i
            in_interval = True
        elif coverage[i] > 0 and in_interval:
            intervals.append(_Interval(rule_id=-1, start=start, end=i))
            zero_rule.occurrences.append(start)
            zero_rule.intervals.append((start, i))
            in_interval = False
    if zero_rule.intervals:
        grammar[-1] = zero_rule

    if not intervals:
        return []

    # Coverage of each interval = mean coverage over its span; sort rare first.
    for iv in intervals:
        iv.cover = float(coverage[iv.start : iv.end].mean())
    intervals.sort(key=lambda iv: iv.cover)

    rng = random.Random(0)
    global_visited = set()
    discords = []

    while len(discords) < num_discords:
        d = _find_best_rra_discord(
            series, win_size, grammar, indexes, intervals, global_visited, rng
        )
        if d.nn_distance < 0:
            break
        discords.append(d)
        mark_start = max(0, d.start - (d.end - d.start))
        mark_end = min(len(series), d.end)
        global_visited.update(range(mark_start, mark_end))

    return discords
