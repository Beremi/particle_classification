import shutil
import subprocess
from pathlib import Path

import pytest


def test_report_builds_with_latexmk():
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")
    report_dir = Path(__file__).resolve().parents[1] / "particle_nn_report_updated"
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "particle_nn_report.tex"],
        cwd=report_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert (report_dir / "particle_nn_report.pdf").exists()
