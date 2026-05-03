from __future__ import annotations

import os
import sys

from setuptools import Extension, setup


def native_extensions() -> list[Extension]:
    try:
        import numpy as np
    except Exception:
        return []

    compile_args = ["-O3"]
    link_args: list[str] = []
    if os.environ.get("PARTICLE_DISABLE_OPENMP") != "1" and sys.platform != "darwin":
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    return [
        Extension(
            "particle_classification._native_clustering",
            ["src/particle_classification/native_clustering.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]


setup(ext_modules=native_extensions())
