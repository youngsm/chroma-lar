"""pyrat passes the database's chroma_use_weights (default False) to simulate."""

import ast
import pathlib

PYRAT = pathlib.Path(__file__).resolve().parents[1] / "pyrat"


def test_simulate_takes_use_weights_from_the_database():
    tree = ast.parse(PYRAT.read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "simulate"]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert ast.unparse(keywords["use_weights"]) == "db.chroma_use_weights"

    import data.chroma

    assert data.chroma.chroma_use_weights is False and "chroma_use_weights" in data.chroma.__exports__
