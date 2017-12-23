"""Converts a normlized timeseries to SAX symbols."""
from saxpy.strfunc import idx2letter
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize


def sax_by_chunking(series, paa_size, alphabet_size=3, z_threshold=0.01):
    """Simple chunking conversion implementation."""
    sax = list()

    paa_rep = paa(znorm(series, z_threshold), paa_size)
    series_len = len(paa_rep)

    alphabet_cuts = cuts_for_asize(alphabet_size)

    for i in range(0, series_len):
        num = paa_rep[i]
        # if teh number below 0, start from the bottom, or else from the top
        if(num >= 0):
            j = alphabet_size - 1
            while ((j > 0) and (alphabet_cuts[j] > num)):
                j = j - 1
            sax.append(idx2letter(j))
        else:
            j = 1
            while (j < alphabet_size and alphabet_cuts[j] < num):
                j = j + 1
            sax.append(idx2letter(j-1))
    return ''.join(sax)
