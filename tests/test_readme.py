"""The README installs with uv and states the LUT rules."""

import pathlib

README = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()


def test_install_is_uv():
    assert "uv sync" in README
    assert "pip install -e git+https://github.com/youngsm/chroma-lite" not in README
    assert "source chroma-lar/env.sh" not in README


def test_lut_rules_are_documented():
    for phrase in ("chroma_use_weights", "CHROMA_TRITON=legacy", "CHROMA_BACKEND=cuda", "18%",
                   "chroma_lar/provenance.py"):
        assert phrase in README
