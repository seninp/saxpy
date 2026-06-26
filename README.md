Time series symbolic discretization with SAX
====
[![Latest PyPI version](https://img.shields.io/pypi/v/saxpy.svg)](https://pypi.python.org/pypi/saxpy)
[![image](http://img.shields.io/:license-gpl2-green.svg)](http://www.gnu.org/licenses/gpl-2.0.html)


This code is released under [GPL v.2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) and implements in Python:
  * Symbolic Aggregate approXimation (SAX) (with z-normalization and PAA) [1]
  * HOT-SAX - a time series anomaly (discord) discovery algorithm [3]
  * RePair - a grammar-inference algorithm over the SAX representation
  * RRA (Rare Rule Anomaly) - a grammar-based, variable-length discord discovery algorithm built on RePair
  * SAX-VSM - a SAX vector-space-model classifier
  * SAX-ENERGY - a multi-dimensional SAX that z-normalizes across data dimensions and aggregates the per-dimension energy. [4]
  * SAX-REPEAT - an extension to SAX for multi-dimensional time-series that performs standard SAX on individual dimensions, then clusters to map multi-dimensional words into strings from the required alphabet size. [4]
  * SAX-INDEPENDENT - a multi-dimensional SAX that SAX-encodes each dimension independently and concatenates the per-dimension words.


Note that all of the library's functionality is also available in [R](https://github.com/jMotif/jmotif-R) and [Java](https://github.com/jMotif/SAX).

#### Cross-implementation alignment

The Python (saxpy), R, and Java implementations are kept aligned: the SAX
stack — z-normalization, PAA, Gaussian breakpoints, symbol assignment — and the
HOT-SAX / brute-force discord search produce the same results across all three
to floating-point precision. The shared conventions are:

  * **z-normalization uses the population standard deviation** (divide by `n`),
    matching the Matrix Profile / MASS convention, so each window has empirical
    variance exactly 1 (the assumption behind SAX's equiprobable breakpoints);
  * **PAA uses fractional segment boundaries** (a sample straddling a segment
    edge is split by overlap);
  * **a value exactly on a breakpoint maps to the symbol above** the cut;
  * **discord search z-normalizes the subsequences** and breaks distance ties by
    the lowest index, so results are reproducible regardless of search order.

The only residual cross-language difference is single-symbol noise for PAA
values that fall within floating-point rounding of a breakpoint — benign and
expected. saxpy is the reference implementation for these conventions.


## References
[1] Lin, J., Keogh, E., Patel, P., and Lonardi, S.,
[*Finding Motifs in Time Series*](http://cs.gmu.edu/~jessica/Lin_motif.pdf),
The 2nd Workshop on Temporal Data Mining, the 8th ACM Int'l Conference on KDD (2002)

[2] Patel, P., Keogh, E., Lin, J., Lonardi, S.,
[*Mining Motifs in Massive Time Series Databases*](http://www.cs.gmu.edu/~jessica/publications/motif_icdm02.pdf),
In Proc. ICDM (2002)

[3] Keogh, E., Lin, J., Fu, A.,
[*HOT SAX: Efficiently finding the most unusual time series subsequence*](http://www.cs.ucr.edu/~eamonn/HOT%20SAX%20%20long-ver.pdf),
In Proc. ICDM (2005)

[4] Mohammad, Y., Nishida T.,
[*Robust learning from demonstrations using multidimensional SAX*](https://ieeexplore.ieee.org/document/6987960),
2014 14th International Conference on Control, Automation and Systems (ICCAS 2014)

## Citing this work

If you are using this implementation for you academic work, please cite our [Grammarviz 2.0 paper](http://link.springer.com/chapter/10.1007/978-3-662-44845-8_37):

[[Citation]](https://raw.githubusercontent.com/jMotif/SAX/master/citation.bib) Senin, P., Lin, J., Wang, X., Oates, T., Gandhi, S., Boedihardjo, A.P., Chen, C., Frankenstein, S., Lerner, M.,  [*GrammarViz 2.0: a tool for grammar-based pattern discovery in time series*](http://csdl.ics.hawaii.edu/techreports/2014/14-06/14-06.pdf), ECML/PKDD Conference, 2014.

SAX in a nutshell
------------
SAX is used to transform a sequence of rational numbers (i.e., a time series) into a sequence of letters (i.e., a string). An illustration of a time series of 128 points converted into the word of 8 letters:

![SAX in a nutshell](https://raw.githubusercontent.com/jMotif/SAX/master/src/resources/sax_transform.png)

As discretization is probably the most used transformation in data mining, SAX has been widely used throughout the field. Find more information about SAX at its authors pages: [SAX overview by Jessica Lin](http://cs.gmu.edu/~jessica/sax.htm), [Eamonn Keogh's SAX page](http://www.cs.ucr.edu/~eamonn/SAX.htm), or at [sax-vsm wiki page](http://jmotif.github.io/sax-vsm_site/morea/algorithm/SAX.html).

Installation
------------
saxpy is published on PyPI, so install it with `pip` (or `uv pip`):

	$ pip install saxpy

saxpy requires Python 3.10+ and depends on `numpy`, `scipy`, and `scikit-learn`.

Development
------------
The project uses [uv](https://docs.astral.sh/uv/) for environment management and packaging
(PEP 621 / `pyproject.toml`, Hatchling build backend). To work on saxpy from a clone:

	$ uv sync                 # create the venv and install saxpy + dev tools
	$ uv run pytest           # run the test suite (unit tests + doctests)
	$ uv run ruff check       # lint
	$ uv run ruff format      # format
	$ uv run mypy saxpy       # type-check

To run the tests across all supported interpreters:

	$ uv run --python 3.10 pytest
	$ uv run --python 3.11 pytest
	$ uv run --python 3.12 pytest

To build the distribution artifacts (`uv build` produces a wheel + sdist in `dist/`),
and enable the pre-commit hooks (ruff + mypy) with `uv run pre-commit install`.

Simple time series to SAX conversion
------------
To convert a time series of an arbitrary length to SAX we need to define the alphabet cuts. Saxpy retrieves cuts for a normal alphabet (we use size 3 here) via `cuts_for_asize` function:

	from saxpy.alphabet import cuts_for_asize
	cuts_for_asize(3)

which yields an array:

	array([      -inf, -0.4307273,  0.4307273])

To convert a time series to letters with SAX we use `ts_to_string` function but not forgetting to z-normalize the input time series (we use Normal alphabet):

	import numpy as np
	from saxpy.znorm import znorm
	from saxpy.sax import ts_to_string
	ts_to_string(znorm(np.array([-2, 0, 2, 0, -1])), cuts_for_asize(3))

this produces a string:

	'abcba'

Time series to SAX conversion with PAA aggregation (by "chunking")
------------
In order to reduce dimensionality further, the PAA (Piecewise Aggregate Approximation) is usually applied prior to SAX:

	import numpy as np
	from saxpy.znorm import znorm
	from saxpy.paa import paa
	from saxpy.sax import ts_to_string

	dat = np.array([-2, 0, 2, 0, -1])
	dat_znorm = znorm(dat)
	dat_paa_3 = paa(dat_znorm, 3)

	ts_to_string(dat_paa_3, cuts_for_asize(3))

and a string with three letters is produced:

	'acb'


Time series to SAX conversion via sliding window
------------
Typically, in order to investigate the input time series structure in order to discover anomalous (i.e., discords) and recurrent (i.e., motifs) patterns we employ time series to SAX conversion via sliding window. Saxpy implements this workflow:

	import numpy as np
	from saxpy.sax import sax_via_window

	dat = np.array([0., 0., 0., 0., 0., -0.270340178359072, -0.367828308500142,
                0.666980581124872, 1.87088147328446, 2.14548907684624,
                -0.480859313143032, -0.72911654245842, -0.490308602315934,
                -0.66152028906509, -0.221049033806403, 0.367003418871239,
                0.631073992586373, 0.0487728723414486, 0.762655178750436,
                0.78574757843331, 0.338239686422963, 0.784206454089066,
                -2.14265084073625, 2.11325193044223, 0.186018356196443,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.519132472499234,
                -2.604783141655, -0.244519550114012, -1.6570790528784,
                3.34184602886343, 2.10361226260999, 1.9796808733979,
                -0.822247322003058, 1.06850578033292, -0.678811824405992,
                0.804225748913681, 0.57363964388698, 0.437113583759113,
                0.437208643628268, 0.989892093383503, 1.76545983424176,
                0.119483882364649, -0.222311941138971, -0.74669456611669,
                -0.0663660879732063, 0., 0., 0., 0., 0.,])

	sax_none = sax_via_window(dat, win_size=6, paa_size=3, alphabet_size=3, nr_strategy=None, znorm_threshold=0.01)

	sax_none

the result is represented as a data structure of resulting words and their respective positions on time series:

	defaultdict(list,
            {'aac': [4, 10, 11, 30, 35],
             'abc': [12, 14, 36, 44],
             'acb': [5, 16, 21, 37, 43],
             'acc': [13, 52, 53, 54],
             'bac': [3, 19, 34, 45, 51],
             'bba': [31],
             'bbb': [15, 18, 20, 22, 25, 26, 27, 28, 29, 41, 42, 46],
             'bbc': [2],
             'bca': [6, 17, 32, 38, 47, 48],
             'caa': [8, 23, 24, 40],
             'cab': [9, 50],
             'cba': [7, 39, 49],
             'cbb': [33],
             'cca': [0, 1]})

`sax_via_window` is parameterised with a sliding window size, desired PAA aggregation, alphabet size, a numerosity reduction strategy, z-normalization threshold, and a SAX type (`'unidim'` for unidimensional SAX (default), and `'repeat'`, `'energy'`, or `'independent'` for multi-dimensional time series):

	def sax_via_window(series, win_size, paa_size, alphabet_size=3,
                   nr_strategy='exact', znorm_threshold=0.01, sax_type='unidim')


Time series discord discovery with HOT-SAX
------------
Saxpy implements HOT-SAX discord discovery algorithm in `find_discords_hotsax` function which can be used as follows:

	import numpy as np
	from saxpy.hotsax import find_discords_hotsax
	from numpy import genfromtxt
	dd = genfromtxt("data/ecg0606_1.csv", delimiter=',')
	discords = find_discords_hotsax(dd)
	discords

and discovers anomalies easily:

	[(430, 5.2790800061718386), (318, 4.1757563573086953)]

The function has a similar parameterization: sliding window size, PAA and alphabet sizes, z-normalization threshold, and a parameter specifying how many discords are desired to be found:

	def find_discords_hotsax(series, win_size=100, num_discords=2, paa_size=3,
                         alphabet_size=3, znorm_threshold=0.01, sax_type='unidim')

Saxpy also provides a brute-force implementation of the discord search if you'd like to verify discords or evaluate the speed-up:

	find_discords_brute_force(series, win_size, num_discords=2,
                              znorm_threshold=0.01)
which can be called as follows:

	discords = find_discords_brute_force(dd[100:500], 100, 4)
	discords

	[(73, 6.198555329625453), (219, 5.5636923991016136)]

Grammar inference with RePair
------------
Saxpy infers a RePair grammar from a space-delimited string of SAX words (or any tokens) via `str_to_repair_grammar`, which returns a dict mapping `rule_id` to a `RepairRule`. Rule 0 (R0) is the compressed top-level string:

	from saxpy.repair import str_to_repair_grammar

	g = str_to_repair_grammar("abc abc cba cba bac xxx abc abc cba cba bac")
	g[0].rule_string          # 'R4 xxx R4'
	g[4].expanded_rule_string # 'abc abc cba cba bac'

Time series discord discovery with RRA
------------
The Rare Rule Anomaly (RRA) algorithm builds a RePair grammar over the SAX representation of a series, derives variable-length subsequences from the grammar rules, and searches them rarest-first. It is exposed as `find_discords_rra`:

	import numpy as np
	from numpy import genfromtxt
	from saxpy.rra import find_discords_rra
	dd = genfromtxt("data/ecg0606_1.csv", delimiter=',')
	discords = find_discords_rra(dd, win_size=100, num_discords=2)
	discords

which returns a list of `RRADiscord` records (variable-length, with start/end positions and the nearest-neighbor distance):

	[RRADiscord(rule_id=76, start=1722, end=1870, length=148, nn_distance=0.024067513139817164),
	 RRADiscord(rule_id=3, start=407, end=508, length=101, nn_distance=0.016651923028144274)]

The function signature is:

	def find_discords_rra(series, win_size, paa_size=3, alphabet_size=3,
	                      nr_strategy="none", znorm_threshold=0.01, num_discords=2)

Time series motif discovery with EMMA
------------
ToDo... (EMMA is not yet implemented in saxpy; it is available in the sibling Java project.)

## Made with Aloha!
![Made with Aloha!](https://raw.githubusercontent.com/GrammarViz2/grammarviz2_src/master/src/resources/assets/aloha.jpg)
