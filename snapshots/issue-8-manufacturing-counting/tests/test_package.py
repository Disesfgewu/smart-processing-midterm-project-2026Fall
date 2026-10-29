import subprocess
import sys

from smart_processing import __version__
from smart_processing.cli import main


def test_version_is_importable():
    assert __version__ == "0.1.0"


def test_cli_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_cli_entry_point_installed():
    result = subprocess.run(
        [sys.executable, "-m", "smart_processing.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in result.stdout
