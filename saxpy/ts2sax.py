"""Converts a normlized timeseries to SAX symbols."""
import numpy as np
from str import idx2letter


def series2sax(series, alphabet_cuts):
    """Simple cnversion implementation."""
    series_len = len(series)
    cuts_len = len(alphabet_cuts)
    sax = list()
    for i in range(0, series_len):
        num = series[i]
        # if teh number below 0, start from the bottom, or else from the top
        if(num >= 0):
            j = cuts_len - 1
            while ((j > 0) and (alphabet_cuts[j] > num)):
                j = j - 1
            sax.append(idx2letter(j))
        else:
            j = 1
            while (j < cuts_len and alphabet_cuts[j] < num):
                j = j + 1
            sax.append(idx2letter(j-1))
    return sax
