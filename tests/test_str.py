"""Testing STR functions."""
import pytest
from saxpy.strfunc import idx2letter


def test_sizing():
    """Test idx to char."""
    assert 'a' == idx2letter(0)
    assert 'h' == idx2letter(7)
    assert 't' == idx2letter(19)

    with pytest.raises(ValueError, match=r'.* idx'):
        idx2letter(-1)

    with pytest.raises(ValueError, match=r'.* idx .*'):
        idx2letter(20)
