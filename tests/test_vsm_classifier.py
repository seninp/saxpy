"""Tests for the SAX-VSM convenience classifier + UCR loader.

Uses the vendored Cylinder-Bell-Funnel (CBF) dataset (the canonical jMotif demo
set, shared with jmotif-R's ``CBF.rda`` and the Java ``sax-vsm_classic`` repo).
"""

import math
import os

import numpy as np

from saxpy.saxvsm import (
    bags_to_tfidf,
    classification_accuracy,
    classify_series,
    load_ucr_data,
    train_tfidf,
)

# Canonical CBF copy already tracked in the repo (same ../-relative pattern as
# tests/test_rra.py). Avoids vendoring a second byte-identical copy under tests/.
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "resources", "data", "cbf")
CBF_TRAIN = os.path.join(DATA_DIR, "CBF_TRAIN")
CBF_TEST = os.path.join(DATA_DIR, "CBF_TEST")

# Discretization parameters from the jmotif-R README §5.0 CBF walkthrough.
WIN, PAA, ALPHA, NR, THR = 60, 8, 6, "exact", 0.01


def test_load_ucr_data_shape():
    """The loader returns {label: [series]} with the documented CBF shape."""
    train = load_ucr_data(CBF_TRAIN)
    test = load_ucr_data(CBF_TEST)
    # CBF: 3 classes, 30 train series, 900 test series, length 128.
    assert sorted(train) == ["1", "2", "3"]
    assert sum(len(v) for v in train.values()) == 30
    assert sum(len(v) for v in test.values()) == 900
    assert all(len(s) == 128 for series in train.values() for s in series)


def test_cbf_classifier_accuracy_golden():
    """Golden: SAX-VSM classifies CBF near-perfectly at the README params.

    All three jMotif impls (saxpy, jmotif-R, Java sax-vsm_classic) score 900/900
    here after the TF*IDF alignment; assert >= 0.99 to stay robust to float
    noise while catching real regressions.
    """
    train = load_ucr_data(CBF_TRAIN)
    test = load_ucr_data(CBF_TEST)
    tfidf = train_tfidf(train, WIN, PAA, ALPHA, NR, THR)
    acc = classification_accuracy(test, tfidf, WIN, PAA, ALPHA, NR, THR)
    assert acc >= 0.99


def test_classify_series_returns_known_label():
    """A trained classifier assigns a test series to one of the train labels."""
    train = load_ucr_data(CBF_TRAIN)
    tfidf = train_tfidf(train, WIN, PAA, ALPHA, NR, THR)
    sample = load_ucr_data(CBF_TEST)["2"][0]
    pred = classify_series(sample, tfidf, WIN, PAA, ALPHA, NR, THR)
    assert pred in {"1", "2", "3"}


def test_classify_series_ambiguous_is_none():
    """A non-overlapping / empty test bag yields an all-equal similarity vector,
    which the classifier reports as ambiguous (None), matching the Java
    convention of a forced misclassification rather than a dict-order winner."""
    train = load_ucr_data(CBF_TRAIN)
    tfidf = train_tfidf(train, WIN, PAA, ALPHA, NR, THR)
    # A flat (constant) series produces a single degenerate word; its bag does
    # not overlap the class vectors -> every class ties at zero similarity.
    flat = np.zeros(128)
    assert classify_series(flat, tfidf, WIN, PAA, ALPHA, NR, THR) is None


def test_tf_scheme_changes_weights_not_signature():
    """bags_to_tfidf accepts tf_scheme; 'log1p' (default) and 'smart' differ, and
    the canonical log1p value matches ln(1 + tf) * ln(N / df)."""
    bags = {
        "a": {"the": 3, "brown": 5, "cow": 2},
        "b": {"the": 3, "green": 2, "hill": 3, "cow": 2},
        "c": {"the": 3, "hill": 2, "cow": 4, "air": 2},
    }
    log1p = bags_to_tfidf(bags)  # default
    smart = bags_to_tfidf(bags, tf_scheme="smart")
    classes = log1p["classes"]
    # "hill" is in b and c only -> df=2, N=3 -> idf=ln(3/2).
    idf = math.log(3 / 2)
    b_idx = classes.index("b")
    assert log1p["vectors"]["hill"][b_idx] == math.log(1 + 3) * idf
    assert smart["vectors"]["hill"][b_idx] == (1 + math.log(3)) * idf
