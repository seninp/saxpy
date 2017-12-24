"""Testing PAA implementation."""
import numpy as np
from saxpy import alphabet
from saxpy import sax


def test_stringing():
    """Test string conversion."""
    # 11: np.array([-np.inf, -1.33517773611894, -0.908457868537385,
    #             -0.604585346583237, -0.348755695517045,
    #             -0.114185294321428, 0.114185294321428, 0.348755695517045,
    #             0.604585346583237, 0.908457868537385, 1.33517773611894]),
    ab = sax.ts_to_string(np.array([-1.33517773611895, -1.33517773611894]),
                          alphabet.cuts_for_asize(11))
    assert 'ab' == ab

    kj = sax.ts_to_string(np.array([1.33517773611895, 1.33517773611894]),
                          alphabet.cuts_for_asize(11))
    assert 'kj' == kj


def test_mindist():
    """Test MINDIST."""
    assert sax.is_mindist_zero('ab', 'ab')
    assert sax.is_mindist_zero('ab', 'bc')
    assert sax.is_mindist_zero('abcd', 'bccc')

    assert 0 == sax.is_mindist_zero('ab', 'bd')
    assert 0 == sax.is_mindist_zero('ab', 'abc')
