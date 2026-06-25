"""Implements VSM."""

import math

import numpy as np

from saxpy.sax import sax_via_window


def series_to_wordbag(
    series, win_size, paa_size, alphabet_size=3, nr_strategy="exact", znorm_threshold=0.01
):
    """VSM implementation."""
    sax = sax_via_window(
        series,
        win_size,
        paa_size,
        alphabet_size=alphabet_size,
        nr_strategy=nr_strategy,
        znorm_threshold=znorm_threshold,
    )

    # convert the dict to a wordbag
    frequencies = {}
    for k, v in sax.items():
        frequencies[k] = len(v)
    return frequencies


def manyseries_to_wordbag(
    series_npmatrix, win_size, paa_size, alphabet_size=3, nr_strategy="exact", znorm_threshold=0.01
):
    """VSM implementation."""
    frequencies = {}

    for row in series_npmatrix:
        tmp_freq = series_to_wordbag(
            np.squeeze(np.asarray(row)),
            win_size,
            paa_size,
            alphabet_size,
            nr_strategy,
            znorm_threshold,
        )
        for k, v in tmp_freq.items():
            if k in frequencies:
                frequencies[k] += v
            else:
                frequencies[k] = v

    return frequencies


def bags_to_tfidf(bags_dict):
    """VSM implementation."""

    # classes
    count_size = len(bags_dict)
    classes = [*bags_dict.copy()]

    # word occurrence frequency counts
    counts = {}

    # compute frequencies
    idx = 0
    for name in classes:
        for word, count in bags_dict[name].items():
            if word in counts:
                counts[word][idx] = count
            else:
                counts[word] = [0] * count_size
                counts[word][idx] = count
        idx = idx + 1

    # compute tf*idf
    tfidf = {}  # the resulting vectors dictionary
    for word, freqs in counts.items():
        # document frequency
        df_counter = 0
        for i in freqs:
            if i > 0:
                df_counter = df_counter + 1

        # if the word is everywhere, dismiss it
        if df_counter == len(freqs):
            continue

        # tf*idf vector
        tf_idf = [0.0] * len(freqs)
        i_idx = 0
        for i in freqs:
            if i != 0:
                tf = np.log(1 + i)
                idf = np.log(len(freqs) / df_counter)
                tf_idf[i_idx] = tf * idf
            i_idx = i_idx + 1

        tfidf[word] = tf_idf

    return {"vectors": tfidf, "classes": classes}


def tfidf_to_vector(tfidf, vector_label):
    """VSM implementation."""
    if vector_label in tfidf["classes"]:
        idx = tfidf["classes"].index(vector_label)
        weight_vec = {}
        for word, weights in tfidf["vectors"].items():
            weight_vec[word] = weights[idx]
        return weight_vec
    else:
        # Unknown label -> empty weight vector (not a [] sentinel, which would
        # crash cosine_measure's `.keys()` access). An empty dict yields a zero
        # norm there, i.e. cosine 0 / maximal distance for an unknown class.
        return {}


def cosine_measure(weight_vec, test_bag):
    """VSM implementation."""
    sumxx, sumxy, sumyy = 0, 0, 0
    for word in set([*weight_vec.copy()]).union([*test_bag.copy()]):
        x, y = 0, 0
        if word in weight_vec:
            x = weight_vec[word]
        if word in test_bag:
            y = test_bag[word]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    # Guard the norm: an empty/non-overlapping test bag or an all-zero class
    # vector (e.g. every word dismissed by idf) makes a norm zero. Treat that as
    # zero cosine -> distance 1.0 under the 1 - cosine convention, rather than
    # crashing the whole classification with ZeroDivisionError.
    denom = math.sqrt(sumxx * sumyy)
    if denom == 0:
        return 0.0
    return sumxy / denom


def cosine_similarity(tfidf, test_bag):
    """Per-class cosine DISTANCE between ``test_bag`` and each class vector.

    NOTE: despite the name this returns cosine *distance* (``1 - cosine``), not
    similarity, so a smaller value means a closer match. ``class_for_bag``
    relies on that by picking the class with the minimum value.
    """
    res = {}
    for cls in tfidf["classes"]:
        res[cls] = 1.0 - cosine_measure(tfidf_to_vector(tfidf, cls), test_bag)

    return res


def class_for_bag(similarity_dict):
    # do i need to take care about equal values?
    return min(similarity_dict, key=lambda x: similarity_dict[x])
