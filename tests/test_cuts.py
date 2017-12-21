"""Testing PAA implementation."""
from saxpy import alphabet


def test_sizing():
    """Test alphabet sizes."""
    for s in range(2, 20):
        assert len(alphabet.cuts_for_asize(s)) == s
