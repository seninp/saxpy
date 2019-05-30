"""Converts a normlized timeseries to SAX symbols."""
from collections import defaultdict
from saxpy.strfunc import idx2letter
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize
import numpy as np
from sklearn.cluster import KMeans


def ts_to_string(series, cuts, sax_type='unidim'):
    """A straightforward num-to-string conversion.

    >>> ts_to_string([[1, 2, -3], [4, 9, -2], [5, 7, -8], [0, 3, -1], [-1, -2, -10]], cuts_for_asize(3), 'repeat')
    'cccba'

    >>> ts_to_string([-1, 0, 1], cuts_for_asize(3))
    'abc'
    """

    series = np.array(series)
    a_size = len(cuts)
    sax = list()

    if sax_type == 'repeat':
        multidim_sax_list = []
        for i in range(series.shape[0]):
            multidim_sax = []
            for j in range(series.shape[1]):
                num = series[i][j]

                # If the number is below 0, start from the bottom, otherwise from the top
                if num >= 0:
                    j = a_size - 1
                    while j > 0 and cuts[j] >= num:
                        j = j - 1
                    multidim_sax.append(j)
                else:
                    j = 1
                    while j < a_size and cuts[j] <= num:
                        j = j + 1
                    multidim_sax.append(j - 1)

            multidim_sax_list.append(multidim_sax)

        # List of all the multidimensional words.
        multidim_sax_list = np.array(multidim_sax_list)

        print 'Clustering %d multi-dimensional letters into %d clusters for final SAX letters...' % (multidim_sax_list.shape[0], a_size)

        # Cluster with k-means++.
        kmeans = KMeans(n_clusters=a_size, random_state=0).fit(multidim_sax_list)

        # Cluster indices in sorted order.
        order = np.lexsort(np.rot90(kmeans.cluster_centers_))

        # Map cluster indices to new SAX letters.
        sax = map(lambda cluster_index: idx2letter(order[cluster_index]), kmeans.predict(multidim_sax_list))

        print 'Mapping complete.'
        
        return ''.join(sax)

    else:
        for i in range(series.shape[0]):
            num = series[i]

            # If the number is below 0, start from the bottom, otherwise from the top
            if num >= 0:
                j = a_size - 1
                while j > 0 and cuts[j] >= num:
                    j = j - 1
                sax.append(idx2letter(j))
            else:
                j = 1
                while j < a_size and cuts[j] <= num:
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


def sax_by_chunking(series, paa_size, alphabet_size=3, znorm_threshold=0.01):
    """Simple chunking conversion implementation."""
    paa_rep = paa(znorm(series, znorm_threshold), paa_size)
    cuts = cuts_for_asize(alphabet_size)
    return ts_to_string(paa_rep, cuts)


def sax_via_window(series, win_size, paa_size, alphabet_size=3,
                   nr_strategy='exact', znorm_threshold=0.01, sax_type='unidim'):
    """Simple via window conversion implementation."""

    # Convert to numpy array.
    series = np.array(series)

    # Check on dimensions.
    assert(len(series.shape) <= 2)

    # Breakpoints.
    cuts = cuts_for_asize(alphabet_size)

    sax = defaultdict(list)
    prev_word = ''

    # Sliding window across time dimension.
    for i in range(0, series.shape[0] - win_size):

        # Subsection starting at this index.
        sub_section = series[i: i + win_size]

        # Z-normalized subsection.
        zn = znorm(sub_section, znorm_threshold)

        # PAA representation of subsection.
        paa_rep = paa(zn, paa_size, sax_type)

        # SAX representation of subsection.
        curr_word = ts_to_string(paa_rep, cuts, sax_type)

        if '' != prev_word:
            if 'exact' == nr_strategy and prev_word == curr_word:
                continue
            elif 'mindist' == nr_strategy and is_mindist_zero(prev_word, curr_word):
                continue

        prev_word = curr_word

        sax[curr_word].append(i)

    return sax
