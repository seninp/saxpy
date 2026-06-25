"""SAX stack implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("saxpy")
except PackageNotFoundError:  # package is not installed (e.g. running from a source checkout)
    __version__ = "0.0.0"

__author__ = "Pavel Senin <seninp@gmail.com>"
__all__ = []
