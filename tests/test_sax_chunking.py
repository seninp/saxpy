"""Testing SAX implementation."""
import numpy as np
from saxpy.sax import sax_by_chunking


def test_chunking():
    """Test SAX by chunking."""
    dat1 = np.array([2.02, 2.33, 2.99, 6.85, 9.2, 8.8, 7.5, 6, 5.85,
                     3.85, 4.85, 3.85, 2.22, 1.45, 1.34])

    dats1_9_7 = "bcggfddba"
    dats1_10_11 = "bcjkjheebb"
    dats1_14_10 = "bbdijjigfeecbb"

    assert dats1_9_7 == sax_by_chunking(dat1, 9, 7)
    assert dats1_10_11 == sax_by_chunking(dat1, 10, 11)
    assert dats1_14_10 == sax_by_chunking(dat1, 14, 10)

    dat2 = np.array([0.5, 1.29, 2.58, 3.83, 3.25, 4.25, 3.83, 5.63, 6.44, 6.25,
                    8.75, 8.83, 3.25, 0.75, 0.72])

    dats2_9_7 = "accdefgda"
    dats2_10_11 = "bcefgijkcb"
    dats2_14_10 = "abdeeffhijjfbb"

    assert dats2_9_7 == sax_by_chunking(dat2, 9, 7)
    assert dats2_10_11 == sax_by_chunking(dat2, 10, 11)
    assert dats2_14_10 == sax_by_chunking(dat2, 14, 10)
