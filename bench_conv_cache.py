# -*- coding: utf-8 -*-
"""Compatibility wrapper for the conv cache benchmark.

This lets you run the benchmark from the repository root with:
python bench_conv_cache.py ...
"""

from pathlib import Path
import runpy


SCRIPT = Path(__file__).resolve().parent / "code" / "tests" / "bench_conv_cache.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")