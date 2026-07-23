"""Bundled sample datasets for pip-installed users."""

from importlib.resources import files


def ecg0606_path():
    """Return the path to the bundled ``ecg0606_1.csv`` PHYSIONET sample."""
    return files("saxpy") / "data" / "ecg0606_1.csv"
