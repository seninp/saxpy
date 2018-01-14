Time series symbolic discretization with SAX
=====

.. image:: https://img.shields.io/pypi/v/saxpy.svg
   :target: https://pypi.python.org/pypi/saxpy
   :alt: Latest PyPI version

.. image:: https://travis-ci.org/seninp/saxpy.png
   :target: https://travis-ci.org/seninp/saxpy
   :alt: Latest Travis CI build status

.. image:: https://codecov.io/gh/seninp/saxpy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/seninp/saxpy

.. image:: http://img.shields.io/:license-gpl2-green.svg
   :target: http://www.gnu.org/licenses/gpl-2.0.html


This code is released under `GPL v.2.0 <https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html>`_ and implements in Python: 
 * Symbolic Aggregate approXimation (i.e., SAX) stack [LIN2002]
 * a simple function for time series motif discovery [PATEL2001]
 * HOT-SAX - a time series anomaly (discord) discovery algorithm [KEOGH2005]

.. [LIN2002] Lin, J., Keogh, E., Patel, P., and Lonardi, S., `Finding Motifs in Time Series <http://cs.gmu.edu/~jessica/Lin_motif.pdf>`_, The 2nd Workshop on Temporal Data Mining, the 8th ACM Int'l Conference on KDD (2002)
.. [PATEL2001] Patel, P., Keogh, E., Lin, J., Lonardi, S., `Mining Motifs in Massive Time Series Databases <http://www.cs.gmu.edu/~jessica/publications/motif_icdm02.pdf>`__, In Proc. ICDM (2002)
.. [KEOGH2005] Keogh, E., Lin, J., Fu, A., `HOT SAX: Efficiently finding the most unusual time series subsequence <http://www.cs.ucr.edu/~eamonn/HOT%20SAX%20%20long-ver.pdf>`__, In Proc. ICDM (2005)

Note that the most of the library's functionality is also available in `R <https://github.com/jMotif/jmotif-R>`__ and `Java <https://github.com/jMotif/SAX>`__


Citing this work:
-----
If you are using this implementation for you academic work, please cite our `Grammarviz 2.0
paper <http://link.springer.com/chapter/10.1007/978-3-662-44845-8_37>`__:

.. [SENIN2014] Senin, P., Lin, J., Wang, X., Oates, T., Gandhi, S., Boedihardjo, A.P., Chen, C., Frankenstein, S., Lerner, M., `GrammarViz 2.0: a tool for grammar-based pattern discovery in time series <http://csdl.ics.hawaii.edu/techreports/2014/14-06/14-06.pdf>`__, ECML/PKDD, 2014.

In a nutshell
-------------
SAX is used to transform a sequence of rational numbers (i.e., a time series) into a sequence of letters (i.e., a string) which is (typically) much shorterthan the input time series. Thus, SAX transform addresses a chief problem in time-series analysis -- the dimensionality curse.

Installation
------------

Requirements
^^^^^^^^^^^^

Compatibility
-------------

Licence
-------
GNU General Public License v2.0

Authors
-------

`saxpy` was written by `Pavel Senin <senin@hawaii.edu>`_.
