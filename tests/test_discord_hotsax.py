"""Testing HOTSAX discord search implementation."""

import random

import numpy as np

import saxpy.hotsax as hotsax
from saxpy.discord import find_discords_brute_force
from saxpy.hotsax import find_discords_hotsax


def test_brute_force():
    """Test HOT-SAX discord discovery."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=",")

    discords = find_discords_hotsax(dd)
    assert 430 == discords[0][0]
    assert 318 == discords[1][0]

    discords = find_discords_hotsax(dd[0:400], 100, 4)
    assert 3 == len(discords)
    assert 173 == discords[0][0]


def test_discord_distance_precision():
    """Pin HOT-SAX discord distances on real data (audit follow-up): a refactor
    that perturbs the distance/early-abandon path is caught here."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=",")
    discords = find_discords_hotsax(dd)
    assert [idx for idx, _ in discords] == [430, 318]
    np.testing.assert_allclose(
        [dist for _, dist in discords],
        [5.279080006171839, 4.175756357308695],
        rtol=0,
        atol=1e-9,
    )


def test_agrees_with_brute_force_on_real_data():
    """HOT-SAX and the brute-force reference must find the same discords (same
    indices, same distances within float noise) on real data -- the
    randomised search order changes neither algorithm's result."""
    dd = np.genfromtxt("data/ecg0606_1.csv", delimiter=",")
    hs = find_discords_hotsax(dd[100:500], win_size=100, num_discords=2)
    bf = find_discords_brute_force(dd[100:500], 100, 2)
    assert [idx for idx, _ in hs] == [idx for idx, _ in bf]
    np.testing.assert_allclose(
        [dist for _, dist in hs],
        [dist for _, dist in bf],
        rtol=0,
        atol=1e-9,
    )


def test_result_is_deterministic_across_seeds():
    """Regression (audit #16): np.random.permutation in the random-search phase
    is unseeded, but the deterministic lowest-index tie-break must keep the
    reported discords stable across RNG states even on tie-heavy data."""
    pattern = np.random.RandomState(0).randn(20)
    series = np.tile(pattern, 15)
    series[150:160] += 6.0
    results = set()
    for seed in range(8):
        random.seed(seed)
        np.random.seed(seed)
        discords = find_discords_hotsax(series, win_size=20, num_discords=1)
        results.add(tuple((idx, round(float(dist), 9)) for idx, dist in discords))
    assert len(results) == 1


def test_param_order_matches_sax_via_window():
    """Regression (audit #31): find_discords_hotsax used to order its size
    params (alphabet_size, paa_size) -- the reverse of sax_via_window
    (paa_size, alphabet_size) -- a silent footgun for positional callers. The
    signature is now (..., paa_size, alphabet_size, ...). Passing them
    positionally must bind paa_size then alphabet_size, matching the keyword
    call."""
    import inspect

    params = list(inspect.signature(find_discords_hotsax).parameters)
    assert params.index("paa_size") < params.index("alphabet_size")

    series = np.random.RandomState(0).randn(300)
    series[100:130] += 8.0
    positional = find_discords_hotsax(series, 30, 1, 4, 5)  # paa_size=4, alphabet_size=5
    keyword = find_discords_hotsax(series, win_size=30, num_discords=1, paa_size=4, alphabet_size=5)
    assert positional == keyword


def test_znorm_threshold_is_forwarded(monkeypatch):
    """Regression (audit #2): find_discords_hotsax used to hardcode
    znorm_threshold=0.01 in its internal sax_via_window call while the distance
    znorms used the caller's value, so the SAX search-order proxy and the
    distance space disagreed for any non-default threshold. The caller's
    threshold must reach sax_via_window unchanged.
    """
    seen = {}
    orig = hotsax.sax_via_window

    def spy(*args, **kwargs):
        seen["znorm_threshold"] = kwargs.get("znorm_threshold")
        return orig(*args, **kwargs)

    monkeypatch.setattr(hotsax, "sax_via_window", spy)

    series = np.random.RandomState(0).randn(200)
    series[100:110] += 8.0
    find_discords_hotsax(series, win_size=30, num_discords=1, znorm_threshold=2.5)

    assert seen["znorm_threshold"] == 2.5
