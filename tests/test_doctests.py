import doctest
import warnings
from sklearn.exceptions import ConvergenceWarning
from saxpy import paa, sax, znorm

def run_doctests(module):
    """Helper to run doctests on a module and fail if any fail."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Number of distinct clusters",
            category=ConvergenceWarning,
            module=r"^sklearn(\.|$)",
        )
        failures, _ = doctest.testmod(
            module,
            optionflags=doctest.NORMALIZE_WHITESPACE
        )
    assert failures == 0, f"Doctests failed in {module.__name__}"

def test_paa_doctests():
    """Test PAA module doctests."""
    run_doctests(paa)

def test_sax_doctests():
    """Test SAX module doctests."""
    run_doctests(sax)

def test_znorm_doctests():
    """Test Z-norm module doctests."""
    run_doctests(znorm)