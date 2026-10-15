"""How a simulation run was made, for the root attributes of the files it writes."""

import json
import os
import subprocess
from importlib import metadata


def _git(path, *args):
    try:
        result = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_state(dist_name, module_file):
    """``(version, commit, dirty)`` of the code ``module_file`` was imported from.

    A git install (pip or uv from a URL) reports the commit recorded in its
    ``direct_url.json``. An editable install or a checkout on ``PYTHONPATH``
    reports the checkout's HEAD, and ``dirty`` 1 when tracked files have local
    changes. A copy installed in site-packages is never matched to a git
    repository that happens to enclose it (a ``.venv`` inside a checkout).
    Unknown values are ``""`` and 0.
    """
    version, info = "", {}
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        dist = None
    if dist is not None:
        version = dist.version
        info = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = info.get("vcs_info", {}).get("commit_id", "")
    if commit:
        return version, commit, 0
    where = os.path.dirname(os.path.abspath(module_file))
    if "site-packages" in where.split(os.sep):
        return version, "", 0
    head = _git(where, "rev-parse", "HEAD")
    if not head:
        return version, "", 0
    changes = _git(where, "status", "--porcelain", "--untracked-files=no")
    return version, head, int(bool(changes))


def _version(dist_name):
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return ""


def _gpu_name():
    try:
        import torch

        return torch.cuda.get_device_name() if torch.cuda.is_available() else ""
    except (ImportError, RuntimeError):
        return ""


def run_attrs(use_weights):
    """The attributes describing this run (backend, options, code revisions, GPU);
    ``use_weights`` is the value passed to ``Simulation.simulate``."""
    import chroma
    import chroma_lar
    from chroma.backend import backend_name

    trichroma_version, trichroma_commit, trichroma_dirty = source_state("trichroma", chroma.__file__)
    lar_version, lar_commit, lar_dirty = source_state("chroma-lar", chroma_lar.__file__)
    return {
        "backend": backend_name(),
        "chroma_triton": os.environ.get("CHROMA_TRITON", ""),
        "chroma_triton_tape": os.environ.get("CHROMA_TRITON_TAPE", ""),
        "trichroma_version": trichroma_version,
        "trichroma_commit": trichroma_commit,
        "trichroma_dirty": trichroma_dirty,
        "chroma_lar_version": lar_version,
        "chroma_lar_commit": lar_commit,
        "chroma_lar_dirty": lar_dirty,
        "use_weights": int(bool(use_weights)),
        "torch_version": _version("torch"),
        "triton_version": _version("triton"),
        "gpu_name": _gpu_name(),
    }


def write_run_attrs(h5file, use_weights):
    """Write :func:`run_attrs` into ``h5file``'s root attributes."""
    for key, value in run_attrs(use_weights).items():
        h5file.attrs[key] = value
