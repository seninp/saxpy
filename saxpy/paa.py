"""Implements PAA."""
import math
import numpy as np


def paa(series, paa_segments):
    """PAA implementation."""
    series_len = len(series)

    # check for the trivial cases
    if series_len == paa_segments:
        paa_series = np.copy(series)
    elif paa_segments == 1:
        paa_series = np.copy(series) / series_len
    else:
        # non-trivial case
        paa_series = np.zeros(paa_segments)
        paa_series_len = len(paa_series)
        # generate list of series break points
        series_break_points = []
        break_point = series_len / paa_segments
        for i in range(paa_segments):
            series_break_points.append(i * break_point)
        # generate paa series
        paa_segment_idx = 0
        for i in range(series_len):
            if paa_segment_idx + 1 < paa_series_len:
                # next i is in new segment
                if i + 1 == math.ceil(series_break_points[paa_segment_idx + 1]):
                    # split ratio for value at break point
                    i_frac = math.modf(series_break_points[paa_segment_idx + 1])[0]
                    j_frac = 1 - i_frac
                    if i_frac != 0:
                        paa_series[paa_segment_idx] += series[i] * i_frac
                        paa_series[paa_segment_idx + 1] += series[i] * j_frac
                    else:
                        paa_series[paa_segment_idx] += series[i]
                # i is first element of new segment 
                elif i == math.ceil(series_break_points[paa_segment_idx + 1]):
                    paa_segment_idx += 1
                    paa_series[paa_segment_idx] += series[i]
                else:
                    paa_series[paa_segment_idx] += series[i]
            else:
                 paa_series[paa_segment_idx] += series[i]
        paa_series = paa_series / break_point
    return paa_series
