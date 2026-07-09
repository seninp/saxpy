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


def _term_frequency(count, tf_scheme):
    """Term-frequency weighting for a single (word, class) raw ``count``.

    Cross-implementation note: SAX-VSM impls historically disagreed on the TF
    transform. As of ``sax-vsm_classic`` 2.0.0, Java, saxpy, and jmotif-R all
    default to ``log1p`` (``ln(1 + tf)``). ``smart`` (``1 + ln(tf)``, the SMART
    ``ltc`` log-TF) remains available for experiments and matches pre-2.0.0 Java
    releases. The schemes are not a uniform scalar of each other, so they can
    change which class wins. The IDF base (``ln`` here) is a uniform per-word
    factor and is cosine-invariant, so it never affects classification.
    """
    if tf_scheme == "log1p":
        return np.log(1 + count)
    if tf_scheme == "smart":
        return 1.0 + np.log(count)
    raise ValueError(f"unknown tf_scheme {tf_scheme!r}; expected 'log1p' or 'smart'")


def bags_to_tfidf(bags_dict, tf_scheme="log1p"):
    """Compute TF*IDF weight vectors for a set of word bags.

    ``tf_scheme`` selects the term-frequency transform (``"log1p"`` =
    ``ln(1 + tf)``, the default across Java/R/Python since sax-vsm 2.0.0;
    ``"smart"`` = ``1 + ln(tf)``, the pre-2.0.0 Java scheme). See
    :func:`_term_frequency`.
    """

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
                tf = _term_frequency(i, tf_scheme)
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


def cosine_distance(tfidf, test_bag):
    """Per-class cosine *distance* (``1 - cosine``) between ``test_bag`` and each
    class vector.

    A smaller value means a closer match, so :func:`class_for_bag` picks the
    class with the minimum value.
    """
    res = {}
    for cls in tfidf["classes"]:
        res[cls] = 1.0 - cosine_measure(tfidf_to_vector(tfidf, cls), test_bag)

    return res


def cosine_similarity(tfidf, test_bag):
    """Deprecated alias for :func:`cosine_distance`.

    Despite the name this returns cosine *distance* (``1 - cosine``), not
    similarity. The function was published under this misleading name; it is
    kept as an alias for backward compatibility and may be removed in a future
    release. Prefer :func:`cosine_distance`.
    """
    return cosine_distance(tfidf, test_bag)


def class_for_bag(distance_dict):
    """Return the label with the smallest cosine distance (the best match).

    ``distance_dict`` is the per-class mapping from :func:`cosine_distance`.
    """
    return min(distance_dict, key=lambda x: distance_dict[x])


# ---------------------------------------------------------------------------
# Convenience classifier layer (train / classify) + UCR data loader.
#
# These compose the building blocks above into a runnable SAX-VSM classifier,
# matching the Java sax-vsm_classic (SAXVSMClassifier) and jmotif-R README §5.0
# walkthroughs. The pipeline is: per-class word bags -> TF*IDF class vectors;
# classify a new series by the largest cosine similarity (smallest 1 - cosine).
# ---------------------------------------------------------------------------


def load_ucr_data(path):
    """Load a UCR/jMotif text dataset into ``{label: [np.ndarray, ...]}``.

    The format (shared by the jMotif Java/R demos) is one series per line:
    a class label in column 0, then the series values, whitespace- or
    comma-separated. Labels are returned as strings (e.g. ``"1"``); a numeric
    label like ``1.0000000e+000`` is normalized to its integer string ``"1"``.
    Mirrors ``net.seninp.util.UCRUtils.readUCRData``.
    """
    data = {}
    with open(path) as fh:
        for line in fh:
            parts = [p for p in line.replace(",", " ").split() if p]
            if not parts:
                continue
            label = parts[0]
            try:
                label = str(int(round(float(label))))
            except ValueError:
                pass
            values = np.array([float(x) for x in parts[1:]])
            data.setdefault(label, []).append(values)
    return data


def train_tfidf(
    labeled_data,
    win_size,
    paa_size,
    alphabet_size=3,
    nr_strategy="exact",
    znorm_threshold=0.01,
    tf_scheme="log1p",
):
    """Train SAX-VSM class vectors from labeled training series.

    ``labeled_data`` maps each class label to a sequence of training series
    (as returned by :func:`load_ucr_data`). Each class becomes one merged word
    bag (:func:`manyseries_to_wordbag`); the bags are then TF*IDF-weighted
    (:func:`bags_to_tfidf`). Returns the tfidf structure ready for
    :func:`classify_series`.
    """
    bags = {
        label: manyseries_to_wordbag(
            np.asarray(series_list),
            win_size,
            paa_size,
            alphabet_size,
            nr_strategy,
            znorm_threshold,
        )
        for label, series_list in labeled_data.items()
    }
    return bags_to_tfidf(bags, tf_scheme=tf_scheme)


def classify_series(
    series,
    tfidf,
    win_size,
    paa_size,
    alphabet_size=3,
    nr_strategy="exact",
    znorm_threshold=0.01,
):
    """Predict the class label of one series against trained TF*IDF vectors.

    The series is converted to a word bag (same SAX transform as training) and
    assigned to the class with the largest cosine similarity. If every class
    ties (e.g. a non-overlapping bag yields an all-zero similarity vector) the
    result is ambiguous and ``None`` is returned -- mirroring the Java
    classifier, which counts an all-equal result as a (forced) misclassification
    rather than letting dict order pick a winner.
    """
    test_bag = series_to_wordbag(
        series, win_size, paa_size, alphabet_size, nr_strategy, znorm_threshold
    )
    distances = cosine_distance(tfidf, test_bag)
    if len(set(distances.values())) <= 1:
        return None
    return class_for_bag(distances)


def classification_accuracy(
    test_data,
    tfidf,
    win_size,
    paa_size,
    alphabet_size=3,
    nr_strategy="exact",
    znorm_threshold=0.01,
):
    """Fraction of ``test_data`` series classified correctly.

    ``test_data`` is a ``{label: [series, ...]}`` mapping like the training
    data. Each series is classified with :func:`classify_series`; an ambiguous
    (``None``) prediction counts as incorrect.
    """
    correct = 0
    total = 0
    for true_label, series_list in test_data.items():
        for series in series_list:
            predicted = classify_series(
                np.squeeze(np.asarray(series)),
                tfidf,
                win_size,
                paa_size,
                alphabet_size,
                nr_strategy,
                znorm_threshold,
            )
            if predicted == true_label:
                correct += 1
            total += 1
    return correct / total if total else 0.0
