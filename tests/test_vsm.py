"""Testing SAX implementation."""

import numpy as np

from saxpy.sax import sax_via_window
from saxpy.saxvsm import (
    bags_to_tfidf,
    cosine_measure,
    cosine_similarity,
    manyseries_to_wordbag,
    series_to_wordbag,
    tfidf_to_vector,
)


def test_series_to_wordbag():
    """Test TS to vector."""
    dat = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.270340178359072,
            -0.367828308500142,
            0.666980581124872,
            1.87088147328446,
            2.14548907684624,
            -0.480859313143032,
            -0.72911654245842,
            -0.490308602315934,
            -0.66152028906509,
            -0.221049033806403,
            0.367003418871239,
            0.631073992586373,
            0.0487728723414486,
            0.762655178750436,
            0.78574757843331,
            0.338239686422963,
            0.784206454089066,
            -2.14265084073625,
            2.11325193044223,
            0.186018356196443,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.519132472499234,
            -2.604783141655,
            -0.244519550114012,
            -1.6570790528784,
            3.34184602886343,
            2.10361226260999,
            1.9796808733979,
            -0.822247322003058,
            1.06850578033292,
            -0.678811824405992,
            0.804225748913681,
            0.57363964388698,
            0.437113583759113,
            0.437208643628268,
            0.989892093383503,
            1.76545983424176,
            0.119483882364649,
            -0.222311941138971,
            -0.74669456611669,
            -0.0663660879732063,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    sax_none = sax_via_window(dat, 6, 3, 3, "none", 0.01)

    wordbag = series_to_wordbag(dat, 6, 3, 3, "none", 0.01)
    wordbag2 = manyseries_to_wordbag(np.array([dat, dat]), 6, 3, 3, "none", 0.01)

    frequencies = {}
    for k, v in sax_none.items():
        frequencies[k] = len(v)

    for k, v in wordbag.items():
        assert v == frequencies[k]

    for k, v in wordbag2.items():
        assert v == frequencies[k] * 2


def test_vsm():
    """Test TF*IDF"""
    bag1 = {"this": 10, "is": 1, "a": 2, "sample": 1}
    bag2 = {"this": 10, "is": 1, "another": 2, "example": 3}
    bags = {"bag1": bag1, "bag2": bag2}

    res = bags_to_tfidf(bags)

    example = np.log(1.0 + 3.0) * np.log(2.0 / 1.0)
    bag2_idx = res["classes"].index("bag2")
    assert res["vectors"]["example"][bag2_idx] == example

    a = np.log(1.0 + 2.0) * np.log(2.0 / 1.0)
    bag1_idx = res["classes"].index("bag1")
    assert res["vectors"]["a"][bag1_idx] == a


def test_cosine_measure_zero_norm():
    """Regression (audit #1): cosine_measure divided by sqrt(sumxx * sumyy) with
    no zero guard, so a non-overlapping test bag (or an all-zero class vector)
    raised ZeroDivisionError and aborted the whole classification. A zero norm
    must now yield cosine 0.0 (-> distance 1.0 under the 1 - cosine convention).
    """
    # No shared words -> sumxy == 0 and one norm is zero.
    assert cosine_measure({"a": 1.0}, {"b": 2.0}) == 0.0

    # Realistic path: bags_to_tfidf dismisses a word present in every class,
    # leaving class "c1" with an all-zero weight vector. cosine_similarity must
    # not crash and must report maximum distance (1.0) for that class.
    bags = {"c1": {"shared": 5}, "c2": {"shared": 5, "only2": 5}}
    tfidf = bags_to_tfidf(bags)
    sim = cosine_similarity(tfidf, {"only2": 3})
    assert sim["c1"] == 1.0


def test_tfidf_to_vector_unknown_label():
    """Regression (audit #20): tfidf_to_vector returned a dict for a known class
    but a [] sentinel for an unknown label, which then crashed cosine_measure's
    `.keys()` access with AttributeError. An unknown label must return an empty
    weight vector (dict) that cosine_measure treats as zero similarity."""
    tfidf = {"classes": ["A", "B"], "vectors": {"abc": [0.5, 0.0]}}

    assert tfidf_to_vector(tfidf, "A") == {"abc": 0.5}

    missing = tfidf_to_vector(tfidf, "NOPE")
    assert missing == {}
    # The empty vector must flow through cosine_measure rather than crash.
    assert cosine_measure(missing, {"abc": 1}) == 0.0


# def test_cosine():
