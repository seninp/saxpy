"""SAX-VSM hyperparameter optimization via the DIRECT global optimizer.

This module selects the SAX discretization parameters -- sliding window size,
PAA size, and alphabet size -- for a SAX-VSM classifier by *minimizing
cross-validation classification error on the TRAINING set*. The test set is
never touched during optimization. The search uses ``scipy.optimize.direct``
(the DIviding RECTangles global optimizer), giving saxpy parity with the Java
``sax-vsm_classic`` (``SAXVSMDirectSampler`` + ``SAXVSMCVErrorFunction``) and the
jmotif-R README s6.0 recipe (``nloptr::directL`` + a hand-written ``cverror``
closure).

The objective :func:`cv_error` rebuilds per-class word bags on each CV fold's
training rows and classifies the held-out rows, mirroring the R README recipe
(the canonical Python-side reference). The Java optimization caches each
series' bag once and *subtracts* held-out words before re-deriving TF-IDF; that
is faster on large training sets but adds a mutable-cache codepath for no
measured benefit here, so we rebuild. A future fast path can swap in a
cached-subtract implementation behind :func:`_fold_bags` without changing the
public surface.

All TF-IDF and cosine machinery is reused from :mod:`saxpy.saxvsm`; this module
adds only the CV objective and the optimizer wiring.
"""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import direct

from saxpy.saxvsm import (
    bags_to_tfidf,
    class_for_bag,
    cosine_distance,
    manyseries_to_wordbag,
    series_to_wordbag,
)

# A continuous (window, paa, alphabet) point -- either a plain sequence or the
# float ndarray ``scipy.optimize.direct`` hands the objective.
Point = Sequence[float] | np.ndarray

# Valid alphabet range: <2 is meaningless, >20 has no letter in
# ``saxpy.strfunc.idx2letter`` (the max supported alphabet is 20).
_MIN_ALPHABET = 2
_MAX_ALPHABET = 20


def _round_params(x: Point) -> tuple[int, int, int]:
    """Round a continuous ``(window, paa, alphabet)`` point to integers.

    DIRECT samples continuous centerpoints, so the objective and the result
    extraction both quantize via this shared helper. ``numpy.rint`` rounds half
    to even, matching R's ``round(..., 0)``; it differs from Java's
    ``Math.round`` (round half up) only on exact ``.5`` boundaries, which the
    DIRECT centerpoints (thirds of the box, e.g. 11.667) do not land on.
    """
    win, paa, alphabet = (int(np.rint(v)) for v in x[:3])
    return win, paa, alphabet


def _make_folds(n_samples: int, nfolds: int | None, random_state: int | None) -> list[np.ndarray]:
    """Partition ``range(n_samples)`` into a list of held-out index arrays.

    ``nfolds is None`` or ``nfolds >= n_samples`` yields leave-one-out CV (each
    sample its own fold), matching R's ``K == m`` case and Java's
    ``holdOutSampleSize == 1`` default. Otherwise the samples are shuffled and
    split into ``nfolds`` roughly equal folds (R's ``cvFolds(type="random")``,
    Java's shuffle of ``samples2go``). ``random_state`` seeds the shuffle for
    reproducibility, matching saxpy's optional-seed convention elsewhere.
    """
    if nfolds is not None and nfolds < 2:
        raise ValueError("nfolds must be >= 2 (a single fold has no held-out test set)")

    if nfolds is None or nfolds >= n_samples:
        # LOOCV: each sample is its own fold.
        return [np.array([i]) for i in range(n_samples)]

    rng = np.random.default_rng(random_state)
    order = rng.permutation(n_samples)
    return [np.asarray(fold) for fold in np.array_split(order, nfolds)]


def _fold_bags(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    train_idx: np.ndarray,
    classes: list,
    win: int,
    paa: int,
    alphabet: int,
    nr_strategy: str,
    znorm_threshold: float,
) -> dict:
    """Build per-class word bags from a fold's training rows.

    Extension point: this rebuilds the bags from scratch (byte-identical to the
    R README recipe). A future fast path can cache each series' bag once and
    subtract the held-out series' raw term counts here -- which is correct only
    *before* the TF=ln(1+tf)/IDF transform -- behind this same signature.
    """
    bags = {}
    for cls in classes:
        cls_rows = train_data[train_idx[train_labels[train_idx] == cls]]
        bags[cls] = manyseries_to_wordbag(
            cls_rows,
            win,
            paa,
            alphabet,
            nr_strategy,
            znorm_threshold,
        )
    return bags


def cv_error(
    train_data,
    train_labels,
    params: Point,
    *,
    nr_strategy: str = "exact",
    znorm_threshold: float = 0.01,
    nfolds: int | None = None,
    random_state: int | None = None,
    tf_scheme: str = "log1p",
) -> float:
    """Mean cross-validation classification error of SAX-VSM on the TRAIN set.

    This is the scalar objective DIRECT minimizes, computed entirely on training
    data (the test set is never touched), mirroring
    ``net.seninp.jmotif.direct.SAXVSMCVErrorFunction.valueAt`` and the jmotif-R
    README s6.0 ``cverror(x)``.

    Parameters
    ----------
    train_data : array-like, shape (n_samples, series_length)
        One training series per row.
    train_labels : array-like, shape (n_samples,)
        Class label per row (any hashable label).
    params : sequence of float, length 3
        Continuous ``(window, paa, alphabet)`` point; rounded to ints internally
        (mirrors Java ``Math.round`` / R ``round``).
    nr_strategy : str, optional
        Numerosity-reduction strategy forwarded to the word-bag builders.
    znorm_threshold : float, optional
        Z-normalization variance threshold forwarded to the word-bag builders.
    nfolds : int or None, optional
        ``None`` (default) or ``>= n_samples`` runs leave-one-out CV; otherwise
        runs ``nfolds`` random folds. Must be ``>= 2`` when given.
    random_state : int or None, optional
        Seed for the fold shuffle (reproducible CV). Default ``None`` keeps the
        historical unseeded behavior, matching saxpy's discord convention.

    Returns
    -------
    float
        A value in ``[0.0, 1.0]``. Returns ``1.0`` immediately when ``paa > win``
        (mirrors Java's early return and avoids ``sax_via_window``'s
        ``ValueError``) or when the rounded alphabet falls outside ``[2, 20]``.
        For LOOCV the error is the global misclassified/n ratio (Java's
        ``missclassified / totalSamples``); for k-fold it is the mean per-fold
        error (R's ``mean(errors)``) -- these coincide only for equal-sized
        folds. The fold ratio formulas match their references, but the
        all-equal-cosine rule below follows Java (force a miss), not R
        (first-tie-wins), so the error VALUE can diverge from R on folds where
        every class ties; the chosen *parameters* are unaffected in practice.

    The ``tf_scheme`` keyword (``"log1p"`` default, or ``"smart"``) selects the
    term-frequency transform used when building the per-fold TF*IDF vectors; see
    :func:`saxpy.saxvsm.bags_to_tfidf`.
    """
    win, paa, alphabet = _round_params(params)

    # Guard before any SAX call: paa>win would crash sax_via_window, and an
    # out-of-range alphabet has no valid letters. Both are penalized, not raised.
    if paa > win:
        return 1.0
    if alphabet < _MIN_ALPHABET or alphabet > _MAX_ALPHABET:
        return 1.0

    data = np.asarray(train_data)
    labels = np.asarray(train_labels)
    n_samples = data.shape[0]
    classes = list(dict.fromkeys(labels.tolist()))

    folds = _make_folds(n_samples, nfolds, random_state)

    total_misclassified = 0
    fold_errors = []
    for test_idx in folds:
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_idx] = True
        train_idx = np.flatnonzero(~test_mask)

        bags = _fold_bags(
            data,
            labels,
            train_idx,
            classes,
            win,
            paa,
            alphabet,
            nr_strategy,
            znorm_threshold,
        )
        tfidf = bags_to_tfidf(bags, tf_scheme=tf_scheme)

        fold_misclassified = 0
        for i in test_idx:
            test_bag = series_to_wordbag(
                np.squeeze(np.asarray(data[i])),
                win,
                paa,
                alphabet,
                nr_strategy,
                znorm_threshold,
            )
            distances = cosine_distance(tfidf, test_bag)
            # Java treats an all-equal-distance result (e.g. every class 1.0 from
            # empty / non-overlapping bags) as a forced misclassification rather
            # than letting the first class win by dict order. Mirror that here for
            # cross-impl alignment, without touching the public ``class_for_bag``.
            if len(set(distances.values())) <= 1:
                predicted = None
            else:
                predicted = class_for_bag(distances)
            if predicted != labels[i]:
                fold_misclassified += 1

        total_misclassified += fold_misclassified
        fold_errors.append(fold_misclassified / len(test_idx))

    if nfolds is None or nfolds >= n_samples:
        # LOOCV: global misclassified/n ratio (Java semantics).
        return total_misclassified / n_samples
    # k-fold: mean of per-fold error ratios (R semantics).
    return float(np.mean(fold_errors))


def optimize_parameters(
    train_data,
    train_labels,
    *,
    bounds: Sequence[tuple[float, float]] = ((10, 120), (2, 60), (2, 12)),
    nr_strategy: str = "exact",
    znorm_threshold: float = 0.01,
    nfolds: int | None = None,
    random_state: int | None = None,
    tf_scheme: str = "log1p",
    max_iter: int = 10,
    max_fun: int | None = None,
    locally_biased: bool = False,
    f_min: float = 0.0,
) -> dict:
    """Select SAX-VSM parameters by minimizing CV error with DIRECT.

    Runs ``scipy.optimize.direct`` on :func:`cv_error` over the continuous
    ``bounds`` box for ``(window, paa, alphabet)``. The objective rounds each
    sampled point to integers, so the search effectively walks an integer grid
    while DIRECT reasons over the continuous box (it returns e.g.
    ``x=[65, 11.667, 3.667]`` on CBF, matching the R README's
    ``65 11.66667 3.666667``).

    DIRECT defaults here mirror the Java/R global (non-biased) search:
    ``locally_biased=False`` and ``f_min=0.0`` so the search can early-terminate
    once CV error reaches its natural floor of zero.

    Selection tie-break: CV error is integer-quantized, so many points tie at the
    minimum. ``scipy.direct`` returns a single ``res.x``, so we ignore it for
    selection and instead scan every evaluated point: among all points at the
    minimum observed error, we pick the one with the SMALLEST window (then the
    natural tuple order on paa, alphabet). This matches README s5.0's documented
    convention and the Java ``SAXVSMDirectSampler`` loop.

    Parameters
    ----------
    bounds : sequence of (lo, hi), length 3
        Continuous box for ``(window, paa, alphabet)``. NOTE: the default window
        upper bound (120) assumes a series length ``>= ~120`` (like CBF, L=128);
        for shorter datasets set the window upper bound below the series length,
        otherwise over-length windows silently yield empty bags (a
        misclassification penalty that steers DIRECT away, not an error).
    max_iter, max_fun, locally_biased, f_min
        Forwarded to ``scipy.optimize.direct`` (as ``maxiter``, ``maxfun``,
        ``locally_biased``, ``f_min``).
    nr_strategy, znorm_threshold, nfolds, random_state, tf_scheme
        Forwarded to :func:`cv_error`.

    Returns
    -------
    dict
        ``{"window_size", "paa_size", "alphabet_size", "cv_error",
        "evaluations", "nfev"}``. ``evaluations`` is the full list of
        ``((w, p, a), err)`` the objective saw (may contain duplicates -- DIRECT
        can revisit rounded points; that does not affect the argmin). The chosen
        params already have the smallest-window tie-break applied.

    Raises
    ------
    ValueError
        If fewer than two unique labels are present (classification is
        undefined; SAX-VSM needs ``>= 2`` classes).
    """
    labels = np.asarray(train_labels)
    if len(set(labels.tolist())) < 2:
        raise ValueError("optimize_parameters needs >= 2 classes in train_labels")

    evaluations: list[tuple[tuple[int, int, int], float]] = []

    def objective(x: np.ndarray) -> float:
        err = cv_error(
            train_data,
            train_labels,
            x,
            nr_strategy=nr_strategy,
            znorm_threshold=znorm_threshold,
            nfolds=nfolds,
            random_state=random_state,
            tf_scheme=tf_scheme,
        )
        evaluations.append((_round_params(x), err))
        return err

    res = direct(
        objective,
        bounds=list(bounds),
        maxiter=max_iter,
        maxfun=max_fun,
        locally_biased=locally_biased,
        f_min=f_min,
    )

    # Smallest-window tie-break over ALL evaluated points (res.x is ignored
    # because the integer-quantized error ties widely).
    min_err = min(err for _, err in evaluations)
    candidates = [params for params, err in evaluations if err == min_err]
    win, paa, alphabet = min(candidates)  # tuple order: window, then paa, then alphabet

    return {
        "window_size": win,
        "paa_size": paa,
        "alphabet_size": alphabet,
        "cv_error": min_err,
        "evaluations": evaluations,
        "nfev": res.nfev,
    }
