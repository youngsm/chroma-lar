"""Provenance attributes of a run."""

import json
import subprocess

import h5py

from chroma_lar import provenance

KEYS = {"backend", "chroma_triton", "chroma_triton_tape", "trichroma_version", "trichroma_commit",
        "trichroma_dirty", "chroma_lar_version", "chroma_lar_commit", "chroma_lar_dirty", "use_weights",
        "torch_version", "triton_version", "gpu_name"}


def _git(path, *args):
    subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def _repo_with_module(tmp_path, rel):
    _git(tmp_path, "init", "-q")
    module = tmp_path / rel
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    return module, head


def test_run_attrs_keys_and_types(monkeypatch):
    monkeypatch.delenv("CHROMA_BACKEND", raising=False)
    monkeypatch.setenv("CHROMA_TRITON", "legacy")
    attrs = provenance.run_attrs(use_weights=False)
    assert set(attrs) == KEYS
    assert all(isinstance(v, (str, int)) for v in attrs.values())
    assert attrs["backend"] == "triton"
    assert attrs["chroma_triton"] == "legacy"
    assert attrs["use_weights"] == 0
    assert attrs["torch_version"].startswith("2.5.1")


def test_run_attrs_without_gpu(monkeypatch):
    monkeypatch.setattr(provenance, "_gpu_name", lambda: "")
    assert provenance.run_attrs(use_weights=False)["gpu_name"] == ""


def test_checkout_reports_head_and_dirty(tmp_path):
    module, head = _repo_with_module(tmp_path, "pkg/__init__.py")
    assert provenance.source_state("no-such-dist", str(module)) == ("", head, 0)
    module.write_text("x = 2\n")
    assert provenance.source_state("no-such-dist", str(module)) == ("", head, 1)


def test_site_packages_copy_is_not_matched_to_enclosing_repo(tmp_path):
    module, _ = _repo_with_module(tmp_path, ".venv/lib/python3.10/site-packages/pkg/__init__.py")
    assert provenance.source_state("no-such-dist", str(module)) == ("", "", 0)


def test_git_install_reports_recorded_commit(monkeypatch, tmp_path):
    class Dist:
        version = "0.6.0"

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({"url": "https://github.com/youngsm/trichroma.git",
                               "vcs_info": {"vcs": "git", "commit_id": "abc123"}})

    monkeypatch.setattr(provenance.metadata, "distribution", lambda name: Dist())
    module = tmp_path / "site-packages" / "chroma" / "__init__.py"
    assert provenance.source_state("trichroma", str(module)) == ("0.6.0", "abc123", 0)


def test_write_run_attrs(tmp_path):
    with h5py.File(tmp_path / "out.h5", "w") as f:
        f.attrs["nphotons_generated"] = 5
        provenance.write_run_attrs(f, True)
    with h5py.File(tmp_path / "out.h5", "r") as f:
        assert KEYS <= set(f.attrs)
        assert f.attrs["nphotons_generated"] == 5
        assert f.attrs["use_weights"] == 1
