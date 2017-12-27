"""Converts a normlized timeseries to SAX symbols."""
from collections import defaultdict
from saxpy.strfunc import idx2letter
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize


def ts_to_string(series, cuts):
    """A straightforward num-to-string conversion."""
    a_size = len(cuts)
    sax = list()
    for i in range(0, len(series)):
        num = series[i]
        # if teh number below 0, start from the bottom, or else from the top
        if(num >= 0):
            j = a_size - 1
            while ((j > 0) and (cuts[j] >= num)):
                j = j - 1
            sax.append(idx2letter(j))
        else:
            j = 1
            while (j < a_size and cuts[j] <= num):
                j = j + 1
            sax.append(idx2letter(j-1))
    return ''.join(sax)


def is_mindist_zero(a, b):
    """Check mindist."""
    if len(a) != len(b):
        return 0
    else:
        for i in range(0, len(b)):
            if abs(ord(a[i]) - ord(b[i])) > 1:
                return 0
    return 1


def sax_by_chunking(series, paa_size, alphabet_size=3, z_threshold=0.01):
    """Simple chunking conversion implementation."""
    paa_rep = paa(znorm(series, z_threshold), paa_size)
    cuts = cuts_for_asize(alphabet_size)
    return ts_to_string(paa_rep, cuts)


def sax_via_window(series, win_size, paa_size, alphabet_size=3,
                   nr_strategy='exact', z_threshold=0.01):
    """Simple via window conversion implementation."""
    cuts = cuts_for_asize(alphabet_size)
    sax = defaultdict(list)

    prev_word = ''

    for i in range(0, len(series) - win_size):

        sub_section = series[i:(i+win_size)]

        zn = znorm(sub_section, z_threshold)

        paa_rep = paa(zn, paa_size)

        curr_word = ts_to_string(paa_rep, cuts)

        if '' != prev_word:
            if 'exact' == nr_strategy and prev_word == curr_word:
                continue
            elif 'mindist' == nr_strategy and\
                    is_mindist_zero(prev_word, curr_word):
                continue

        prev_word = curr_word

        sax[curr_word].append(i)

    return sax
