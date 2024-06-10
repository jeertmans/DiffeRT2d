import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    from tomllib import load
else:
    from tomli import load  # type: ignore[reportMissingImports]

from differt2d.__version__ import __version__


def test_version():
    got = __version__
    pyproject = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject, "rb") as file:
        expected = load(file)["tool"]["bumpversion"]["current_version"]

    assert got == expected
