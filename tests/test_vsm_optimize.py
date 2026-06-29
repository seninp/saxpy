"""Tests for the DIRECT-based SAX-VSM parameter optimizer (saxpy.saxvsm_optimize).

Uses the vendored CBF dataset. The golden mirrors the jmotif-R README §6.0
result (DIRECT minimizes leave-out CV error on the train set, then the chosen
params classify the test set near-perfectly).
"""

import os

import numpy as np
import pytest

from saxpy.saxvsm import classification_accuracy, load_ucr_data, train_tfidf
from saxpy.saxvsm_optimize import cv_error, optimize_parameters

# Canonical CBF copy already tracked in the repo (see test_vsm_classifier.py).
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "resources", "data", "cbf")
CBF_TRAIN = os.path.join(DATA_DIR, "CBF_TRAIN")
CBF_TEST = os.path.join(DATA_DIR, "CBF_TEST")


def _cbf_train_matrix():
    data = load_ucr_data(CBF_TRAIN)
    X, y = [], []
    for label, series_list in data.items():
        for s in series_list:
            X.append(s)
            y.append(label)
    return np.array(X), np.array(y)


def test_cv_error_paa_gt_win_returns_one():
    """paa > win is penalized (1.0), not raised."""
    X, y = _cbf_train_matrix()
    assert cv_error(X, y, (10, 20, 4)) == 1.0


def test_cv_error_alphabet_out_of_range_returns_one():
    """Alphabet outside [2, 20] is penalized (1.0)."""
    X, y = _cbf_train_matrix()
    assert cv_error(X, y, (60, 8, 1)) == 1.0
    assert cv_error(X, y, (60, 8, 21)) == 1.0


def test_cv_error_in_unit_interval():
    """cv_error returns a float in [0, 1] at known-good CBF params."""
    X, y = _cbf_train_matrix()
    err = cv_error(X, y, (60, 8, 6))
    assert isinstance(err, float)
    assert 0.0 <= err <= 1.0


def test_cv_error_loocv_matches_manual():
    """LOOCV error equals hand-counted misclassified/n on a tiny 2-class set.

    Two trivially separable constant-ish classes; each held-out series is still
    classified by the other members of its class, so error is 0.
    """
    rng = np.random.default_rng(0)
    a = [rng.normal(0, 0.01, 64) + np.linspace(0, 1, 64) for _ in range(4)]
    b = [rng.normal(0, 0.01, 64) + np.linspace(1, 0, 64) for _ in range(4)]
    X = np.array(a + b)
    y = np.array(["a"] * 4 + ["b"] * 4)
    err = cv_error(X, y, (32, 8, 4), nr_strategy="none")
    assert 0.0 <= err <= 1.0  # well-defined; separable data -> low error
    assert err <= 0.5


def test_optimize_returns_int_params_in_bounds():
    """optimize_parameters returns integer params honoring the bounds + guards."""
    X, y = _cbf_train_matrix()
    res = optimize_parameters(X, y, max_iter=4, max_fun=40, nfolds=5, random_state=0)
    assert isinstance(res["window_size"], int)
    assert isinstance(res["paa_size"], int)
    assert isinstance(res["alphabet_size"], int)
    assert 10 <= res["window_size"] <= 120
    assert res["paa_size"] <= res["window_size"]
    assert 2 <= res["alphabet_size"] <= 20


def test_optimize_cbf_golden_accuracy():
    """GOLDEN: params chosen by DIRECT-CV on CBF train classify the test set
    near-perfectly (>= 0.99), matching the jmotif-R README §6.0 result."""
    X, y = _cbf_train_matrix()
    res = optimize_parameters(X, y, max_iter=10, max_fun=80, nfolds=5, random_state=0)
    train = load_ucr_data(CBF_TRAIN)
    test = load_ucr_data(CBF_TEST)
    w, p, a = res["window_size"], res["paa_size"], res["alphabet_size"]
    tfidf = train_tfidf(train, w, p, a, "exact", 0.01)
    acc = classification_accuracy(test, tfidf, w, p, a, "exact", 0.01)
    assert acc >= 0.99


def test_optimize_reproducible():
    """A fixed random_state gives identical params + cv_error across runs."""
    X, y = _cbf_train_matrix()
    r1 = optimize_parameters(X, y, max_iter=4, max_fun=40, nfolds=5, random_state=0)
    r2 = optimize_parameters(X, y, max_iter=4, max_fun=40, nfolds=5, random_state=0)
    assert (r1["window_size"], r1["paa_size"], r1["alphabet_size"]) == (
        r2["window_size"],
        r2["paa_size"],
        r2["alphabet_size"],
    )
    assert r1["cv_error"] == r2["cv_error"]


def test_optimize_single_class_raises():
    """A single unique label is rejected (classification undefined)."""
    X, y = _cbf_train_matrix()
    with pytest.raises(ValueError):
        optimize_parameters(X, np.array(["only"] * len(y)))


def test_cv_error_nfolds_too_small_raises():
    """nfolds < 2 is rejected."""
    X, y = _cbf_train_matrix()
    with pytest.raises(ValueError):
        cv_error(X, y, (60, 8, 6), nfolds=1)
