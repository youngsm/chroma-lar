"""Macros compile and pick the photon source that matches the backend."""

import ast
import os
import pathlib
import py_compile
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MACROS = sorted((ROOT / "macros").glob("*.py"))


@pytest.mark.parametrize("path", MACROS, ids=lambda p: p.name)
def test_macro_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "out.pyc"), doraise=True)


@pytest.mark.parametrize("name", ["waveform_map_pyrat.py", "waveform_map_pyrat_quantile.py"])
def test_gpu_sources_come_from_chroma_triton(name):
    tree = ast.parse((ROOT / "macros" / name).read_text())
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "chroma.triton" in modules
    assert "trichroma" not in modules


def test_sources_unavailable_on_cuda_backend():
    env = dict(os.environ, CHROMA_BACKEND="cuda")
    code = "from chroma.triton import sources; print(sources.available())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"  # chroma may print a warning first
