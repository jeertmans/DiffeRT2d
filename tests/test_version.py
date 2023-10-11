from pathlib import Path

from rtoml import load

from differt2d.__version__ import __version__


def test_version():
    got = __version__
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    expected = load(pyproject)["tool"]["poetry"]["version"]
    assert got == expected
