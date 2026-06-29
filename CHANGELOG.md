# Changelog

All notable changes to saxpy are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-06

First modern PyPI release. (The only artifact previously on PyPI was the legacy
`1.0.1.dev167` pbr snapshot; 2.0.0 supersedes it.) This release modernizes the
build, adds a grammar layer and a SAX-VSM classifier, and aligns results with the
sibling jMotif implementations (R and Java).

### Added
- **SAX-VSM time-series classification.** A classifier layer in `saxpy.saxvsm`
  (`load_ucr_data`, `train_tfidf`, `classify_series`, `classification_accuracy`)
  built on the existing word-bag / TF·IDF primitives. Verified accuracy 1.0 on the
  Cylinder-Bell-Funnel demo set.
- **DIRECT cross-validation parameter optimizer** (`saxpy.saxvsm_optimize`):
  `optimize_parameters` selects window / PAA / alphabet by minimizing leave-out CV
  error on the training set via `scipy.optimize.direct` (mirrors the jmotif-R and
  Java DiRect recipes). Deterministic with `random_state`; `cv_error` exposes the
  objective.
- **Configurable TF·IDF weighting.** `bags_to_tfidf(tf_scheme=...)` — `"log1p"`
  (default, `ln(1 + tf)`) or `"smart"` (`1 + ln(tf)`).
- **Grammar layer.** RePair grammar inference (`saxpy.repair.str_to_repair_grammar`)
  and RRA variable-length discord discovery (`saxpy.rra.find_discords_rra`), ported
  from the C++/Java implementations.
- **Reproducibility.** Optional `random_state` for HOT-SAX, brute-force, and RRA
  discord search, so results are reproducible across runs and across the
  Python / R / Java implementations.
- `cosine_distance` — correctly-named accessor for the per-class cosine distance.

### Changed
- **Cross-implementation alignment.** The TF·IDF weighting is standardized to
  `log1p` (`ln(1 + tf) · ln(N / df)`) across saxpy, jmotif-R, and the Java
  `sax-vsm_classic`; SAX-VSM is verified 3-way identical on CBF (900/900).
- **RRA discord search**: skip degenerate sub-`paa_size` zero-coverage intervals;
  order candidates by rule frequency (matches GrammarViz).
- **Packaging & tooling**: migrated from pbr to Hatchling, adopted `uv` +
  `pyproject.toml`, ruff + mypy gates, and a 3.10/3.11/3.12 × linux/macOS/windows CI
  matrix. The version is single-sourced from `pyproject.toml`.
- **Minimum Python is now 3.10.**

### Performance
- Brute-force discord search: vectorized the nearest-neighbour distance (one
  numpy pass per candidate instead of a per-neighbour pure-Python early-abandon
  loop), ~90× faster on a full ECG series with identical discords. It now
  computes every candidate's exact NN, so it no longer depends on visit order
  (`random_state` is retained but inert).
- RePair: bucketed (count-indexed) priority queue (Larsson–Moffat), ~2.8× faster on
  the Python side.
- RRA: cached reference z-norm + lightweight 1-D z-norm (~3–6× faster).
- PAA: vectorized (`reshape` / `repeat`-and-mean).

### Fixed
- Adversarial-audit hardening across the SAX / discord / VSM core (divide-by-zero
  guards, on-breakpoint symbol canonicalization, PAA boundary and up-sampling
  validation, deterministic lowest-index discord tie-break, and more).
- RRA `_paa2` float-boundary correctness, and z-normalized discord distance to match
  HOT-SAX / the paper.
- Packaging hygiene: an explicit sdist allowlist, so the published source
  distribution ships only the package and user-facing docs (no dev/internal files
  or test datasets).

### Deprecated
- `cosine_similarity` is a misnomer — it returns cosine *distance* (`1 - cosine`),
  not similarity. It is kept as an alias for `cosine_distance` and may be removed in
  a future release.

### Docs
- README rewritten as the user-facing tutorial: runnable examples across the full
  stack, a SAX-VSM section, the cross-implementation alignment notes, and algorithm
  performance tables.
