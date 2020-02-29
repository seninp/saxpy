"""Converts a normlized timeseries to SAX symbols."""
from collections import defaultdict
from saxpy.strfunc import idx2letter
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize
import numpy as np
from sklearn.cluster import KMeans


# For SAX-REPEAT.
def get_sax_list(series, cuts):
    """
    >>> get_sax_list([[1, 2, -3], [4, 9, -2], [5, 7, -8], [0, 3, -1], [-1, -2, -10]], cuts_for_asize(3))
    [[2, 2, 0], [2, 2, 0], [2, 2, 0], [1, 2, 0], [0, 0, 0]]
    """

    series = np.array(series)
    a_size = len(cuts)
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

    return multidim_sax_list


def ts_to_string(series, cuts):
    """A straightforward num-to-string conversion.

    >>> ts_to_string([-1, 0, 1], cuts_for_asize(3))
    'abc'

    >>> ts_to_string([1, -1, 1], cuts_for_asize(3))
    'cac'
    """

    series = np.array(series)
    a_size = len(cuts)
    sax = list()

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
    """Simple via window conversion implementation.

    # SAX-ENERGY
    >>> sax_via_window([[1, 2, 3], [4, 5, 6]], win_size=1, paa_size=3, sax_type='energy', nr_strategy=None)['abc']
    [0, 1]

    >>> sax_via_window([[1, 2, 3, 4], [4, 5, 6, 7]], win_size=1, paa_size=4, sax_type='energy', nr_strategy=None)['aacc']
    [0, 1]

    >>> sax_via_window([[1, 2, 3, 4], [4, 5, 6, 7]], win_size=2, paa_size=4, sax_type='energy', nr_strategy=None)['aaccaacc']
    [0]

    # SAX-REPEAT
    >>> sax_via_window([[1, 2, 3], [4, 5, 6], [7, 8, 9]], win_size=2, paa_size=2, sax_type='repeat', nr_strategy=None)['ab']
    [0, 1]

    >>> sax_via_window([[1, 2, 3], [4, 5, 6], [7, 8, 9]], win_size=1, paa_size=1, sax_type='repeat', nr_strategy=None)['a']
    [0, 1, 2]

    # SAX-INDEPENDENT
    >>> sax_via_window([[1, 2, 3, 4], [4, 5, 6, 7]], win_size=2, paa_size=2, sax_type='independent', nr_strategy=None)['acacacac']
    [0]

    >>> sax_via_window([[1, 2], [4, 5], [7, 8]], win_size=2, paa_size=2, sax_type='independent', nr_strategy=None)['acac']
    [0, 1]

    >>> sax_via_window([[1, 2], [4, 8], [7, 5]], win_size=2, paa_size=2, sax_type='independent', nr_strategy=None)['acac']
    [0]

    >>> sax_via_window([[1, 2], [4, 8], [7, 5]], win_size=2, paa_size=2, sax_type='independent', nr_strategy=None)['acca']
    [1]

    """

    # Convert to numpy array.
    series = np.array(series)

    # Check on dimensions.
    if len(series.shape) > 2:
        raise ValueError('Please reshape time-series to stack dimensions along the 2nd dimension, so that the array shape is a 2-tuple.')

    # PAA size is the length of the PAA sequence.
    if sax_type != 'energy' and paa_size > win_size:
        raise ValueError('PAA size cannot be greater than the window size.')

    if sax_type == 'energy' and len(series.shape) == 1:
        raise ValueError('Must pass a multidimensional time-series to SAX-ENERGY.')

    # Breakpoints.
    cuts = cuts_for_asize(alphabet_size)

    # Dictionary mapping SAX words to indices.
    sax = defaultdict(list)

    if sax_type == 'repeat':
        # Maps indices to multi-dimensional SAX words.
        multidim_sax_dict = []

        # List of all the multi-dimensional SAX words.
        multidim_sax_list = []

        # Sliding window across time dimension.
        for i in range(series.shape[0] - win_size + 1):

            # Subsection starting at this index.
            sub_section = series[i: i + win_size]

            # Z-normalized subsection.
            if win_size == 1:
                zn = sub_section
            else:
                zn = znorm(sub_section, znorm_threshold)

            # PAA representation of subsection.
            paa_rep = paa(zn, paa_size, 'repeat')

            # SAX representation of subsection, but in terms of multi-dimensional vectors.
            multidim_sax = get_sax_list(paa_rep, cuts)

            # Update data-structures.
            multidim_sax_dict.append(multidim_sax)
            multidim_sax_list.extend(multidim_sax)

        # Cluster with k-means++.
        kmeans = KMeans(n_clusters=alphabet_size, random_state=0).fit(multidim_sax_list)

        # Cluster indices in sorted order.
        order = np.lexsort(np.rot90(kmeans.cluster_centers_))

        # Sliding window across time dimension.
        prev_word = ''
        for i in range(series.shape[0] - win_size + 1):

            # Map cluster indices to new SAX letters.
            curr_word_list = map(lambda cluster_index: idx2letter(order[cluster_index]), kmeans.predict(multidim_sax_dict[i]))
            curr_word = ''.join(curr_word_list)

            if '' != prev_word:
                if 'exact' == nr_strategy and prev_word == curr_word:
                    continue
                elif 'mindist' == nr_strategy and is_mindist_zero(prev_word, curr_word):
                    continue

            prev_word = curr_word

            sax[curr_word].append(i)

    else:
        # Sliding window across time dimension.
        prev_word = ''
        for i in range(series.shape[0] - win_size + 1):

            # Subsection starting at this index.
            sub_section = series[i: i + win_size]

            if sax_type == 'energy':
                curr_word = ''
                for energy_dist in sub_section:
                    # Normalize energy distribution.
                    energy_zn = znorm(energy_dist, znorm_threshold)

                    # PAA representation of energy distribution.
                    paa_rep = paa(energy_zn, paa_size, 'unidim')
                    # paa_rep = energy_zn

                    # SAX representation of the energy distribution.
                    energy_word = ts_to_string(paa_rep, cuts)

                    # Add to current word.
                    curr_word += energy_word

            elif sax_type == 'independent':
                curr_word = ''
                for dim in range(sub_section.shape[1]):
                    # Obtain the subsequence restricted to one dimension.
                    one_dimension_sub_section = sub_section[:, dim]

                    # Z-normalized subsection.
                    zn = znorm(one_dimension_sub_section, znorm_threshold)

                    # PAA representation of subsection.
                    paa_rep = paa(zn, paa_size, 'unidim')

                    # Get the SAX word - just a unidimensional SAX.
                    one_dim_word = ts_to_string(paa_rep, cuts)

                    # Add this dimensions' representation to the overall SAX word.
                    curr_word += one_dim_word

            else:
                # Z-normalized subsection.
                zn = znorm(sub_section, znorm_threshold)

                # PAA representation of subsection.
                paa_rep = paa(zn, paa_size, sax_type)

                # SAX representation of subsection.
                curr_word = ts_to_string(paa_rep, cuts)

            if '' != prev_word:
                if 'exact' == nr_strategy and prev_word == curr_word:
                    continue
                elif 'mindist' == nr_strategy and is_mindist_zero(prev_word, curr_word):
                    continue

            prev_word = curr_word

            sax[curr_word].append(i)

    return sax

