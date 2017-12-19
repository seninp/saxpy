"""Testing PAA implementation."""
from saxpy import alphabet2cuts


def test_sizing():
    """Test alphabet sizes."""
    for s in range(2, 20):
        assert len(alphabet2cuts.cuts_for_asize(s)) == s
