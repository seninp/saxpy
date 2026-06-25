"""Testing PAA implementation."""

import numpy as np

from saxpy import alphabet, sax


def test_stringing():
    """Test string conversion."""
    # 11: np.array([-np.inf, -1.33517773611894, -0.908457868537385,
    #             -0.604585346583237, -0.348755695517045,
    #             -0.114185294321428, 0.114185294321428, 0.348755695517045,
    #             0.604585346583237, 0.908457868537385, 1.33517773611894]),

    print(alphabet.cuts_for_asize(11))

    ab = sax.ts_to_string(
        np.array([-1.33517773611894, -1.33517773611893]), alphabet.cuts_for_asize(11)
    )
    assert "ab" == ab

    kj = sax.ts_to_string(np.array([1.335177736119, 1.3351777361]), alphabet.cuts_for_asize(11))
    assert "kj" == kj


def test_breakpoint_assignment_is_canonical():
    """Regression (audit #4): a value lying exactly on a breakpoint must map to
    the symbol ABOVE it (canonical Lin-2003 / jmotif convention). The old
    sign-based inward walk handled this only for negative breakpoints, mapping
    values on a positive breakpoint one symbol too low. searchsorted fixes it
    symmetrically.
    """
    cuts = alphabet.cuts_for_asize(5)  # [-inf, -0.8416, -0.2533, 0.2533, 0.8416]

    # Exactly on a positive breakpoint -> the symbol above it.
    assert "d" == sax.ts_to_string(np.array([cuts[3]]), cuts)
    assert "e" == sax.ts_to_string(np.array([cuts[4]]), cuts)
    # Exactly on a negative breakpoint (already correct before the fix).
    assert "b" == sax.ts_to_string(np.array([cuts[1]]), cuts)
    # Zero stays put, and ts_to_string / get_sax_list agree symbol-for-symbol.
    assert "c" == sax.ts_to_string(np.array([0.0]), cuts)
    assert [[3]] == sax.get_sax_list([[cuts[3]]], cuts)


def test_mindist():
    """Test MINDIST."""
    assert sax.is_mindist_zero("ab", "ab")
    assert sax.is_mindist_zero("ab", "bc")
    assert sax.is_mindist_zero("abcd", "bccc")

    assert 0 == sax.is_mindist_zero("ab", "bd")
    assert 0 == sax.is_mindist_zero("ab", "abc")
