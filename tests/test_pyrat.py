"""pyrat's event loop simulates unweighted photons (LUTs count photons)."""

import ast
import pathlib

PYRAT = pathlib.Path(__file__).resolve().parents[1] / "pyrat"


def test_simulate_is_called_unweighted():
    tree = ast.parse(PYRAT.read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "simulate"]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "use_weights" in keywords
    assert isinstance(keywords["use_weights"], ast.Constant) and keywords["use_weights"].value is False
