import shutil
import subprocess
from pathlib import Path

import pytest


def test_report_builds_with_latexmk(tmp_path: Path):
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")
    report_dir = (
        Path(__file__).resolve().parents[1]
        / "experimental_notes"
        / "background"
        / "particle_nn_report_updated"
    )
    build_dir = tmp_path / "particle_nn_report"
    shutil.copytree(report_dir, build_dir)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "particle_nn_report.tex"],
        cwd=build_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert (build_dir / "particle_nn_report.pdf").exists()
