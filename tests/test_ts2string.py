"""Testing PAA implementation."""
import numpy as np
from saxpy import alphabet
from saxpy import sax


def test_stringing():
    """Test string conversion."""
    # 11: np.array([-inf, -1.33517774, -0.90845787, -0.60458535,
    #               -0.3487557, -0.11418529, 0.11418529, 0.3487557,
    #               0.60458535, 0.90845787, 1.33517774]),

    ab = sax.ts_to_string(np.array([-1.33517775 , -1.33517773]),
                          alphabet.cuts_for_asize(11))
    assert 'ab' == ab

    kj = sax.ts_to_string(np.array([1.33517775, 1.33517773]),
                          alphabet.cuts_for_asize(11))
    assert 'kj' == kj

    # Test to handel cuts of size 26
    print(alphabet.cuts_for_asize(26))

    yz = sax.ts_to_string(np.array([1.76882503, 1.76882505]),
                          alphabet.cuts_for_asize(26))
    
    assert 'yz' == yz
    


def test_mindist():
    """Test MINDIST."""
    assert sax.is_mindist_zero('ab', 'ab')
    assert sax.is_mindist_zero('ab', 'bc')
    assert sax.is_mindist_zero('abcd', 'bccc')

    assert 0 == sax.is_mindist_zero('ab', 'bd')
    assert 0 == sax.is_mindist_zero('ab', 'abc')
