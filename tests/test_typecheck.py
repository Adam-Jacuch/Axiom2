import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_autocomplete_type_contract():
    pyright = shutil.which("pyright")
    adjacent_pyright = Path(sys.executable).with_name("pyright")
    if pyright is None and adjacent_pyright.exists():
        pyright = str(adjacent_pyright)
    if pyright is None:
        pytest.skip("pyright is not installed")

    contract = Path(__file__).parent / "typecheck" / "autocomplete_contract.py"
    result = subprocess.run(
        [pyright, "--pythonpath", sys.executable, str(contract)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
