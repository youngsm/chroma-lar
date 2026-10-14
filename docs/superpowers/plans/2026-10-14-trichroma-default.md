# TriChroma as chroma-lar's Default Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** chroma-lar installs with uv in minutes (no PyCUDA), runs on TriChroma by default, keeps LUT generation unweighted, records each LUT's provenance, and is validated against CUDA Chroma on turing and ampere GPUs.

**Architecture:** `pyproject.toml` declares the merged `trichroma` distribution (Chroma + `chroma.triton`) as a git dependency; Triton is chroma's own default, so chroma-lar sets nothing. A small `chroma_lar.provenance` module writes run metadata into LUT files; `chroma_lar.validation` compares two waveform-map runs statistically. Container recipes install with uv from the base PyTorch image.

**Tech Stack:** Python 3.10, setuptools, uv, pytest, h5py, PyTorch 2.5.1 + Triton 3.1.0, TriChroma (`chroma`, `chroma.triton`), photonlib, apptainer, Slurm on S3DF.

**Spec:** `/sdf/data/neutrino/youngsam/chroma-lar-trichroma/docs/superpowers/specs/2026-10-14-trichroma-default-design.md`

**Prerequisite:** the merge plan (`/sdf/data/neutrino/youngsam/trichroma-merge/docs/superpowers/plans/2026-10-14-merge-trichroma.md`) through its Task 9, so that `git+https://github.com/youngsm/chroma-lite.git@merge-trichroma` resolves.

## Global Constraints

- Work in the worktree `/sdf/data/neutrino/youngsam/chroma-lar-trichroma`, branch `trichroma-default`. Everything built lives under `/sdf/data/neutrino/youngsam/`, never `/sdf/group/`.
- Commits use the user's git identity, with no Claude attribution or `Co-Authored-By` lines.
- Environment for every shell:
  ```bash
  export UV_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/uv
  export UV_PROJECT_ENVIRONMENT=/sdf/data/neutrino/youngsam/envs/chroma-lar-trichroma
  export TRITON_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/triton
  export APPTAINER_CACHEDIR=/sdf/data/neutrino/youngsam/.apptainer_cache
  export APPTAINER_TMPDIR=/sdf/data/neutrino/youngsam/tmp/apptainer
  ```
- Dependency (until the GitHub rename): `trichroma @ git+https://github.com/youngsm/chroma-lite.git@merge-trichroma`; after the rename (Task 13): `trichroma @ git+https://github.com/youngsm/trichroma.git`. Extra `cuda` = `trichroma[cuda]` at the same URL.
- `photonlib @ git+https://github.com/CIDeR-ML/photonlib.git`, `torch==2.5.1`, plus numpy, scipy, h5py, trimesh, PyYAML, pydantic, rich, tqdm, matplotlib, fire. `requires-python = ">=3.10"`.
- chroma-lar sets no `CHROMA_BACKEND`; Triton is chroma's default. `CHROMA_BACKEND=cuda` (with the `cuda` extra) runs CUDA Chroma.
- `pyrat` calls `sim.simulate(..., use_weights=False)`; there is no switch for weighted mode.
- Provenance keys, exactly: `backend`, `chroma_triton`, `chroma_triton_tape`, `trichroma_version`, `trichroma_commit`, `trichroma_dirty`, `chroma_lar_version`, `chroma_lar_commit`, `chroma_lar_dirty`, `use_weights`, `torch_version`, `triton_version`, `gpu_name`. Values are `str` or `int`.
- Waveform-map macros that write provenance: `macros/waveform_map_pyrat.py`, `macros/waveform_map_pyrat_32bit.py`, `macros/waveform_map_pyrat_quantile.py`.
- GPU jobs: `srun --account=neutrino:cider-nu --partition=ampere|turing --gpus=1 --cpus-per-gpu=4 --mem=48G --time=3:00:00`.
- Pushing the feature branch `trichroma-default` is pre-approved. Changing the dependency URL after the rename needs the user's yes at that step.

## Review Focus

- A uv environment that lives inside a git checkout must not report that checkout's commit as the installed TriChroma's. Test: Task 4 `test_site_packages_copy_is_not_matched_to_enclosing_repo`.
- With `CHROMA_BACKEND=cuda`, the macros must draw numpy photons, not hand CUDA tensors to CUDA Chroma. Test: Task 2 `test_sources_unavailable_on_cuda_backend`.
- Comparing two runs whose voxel sets differ must fail and name the missing voxels, not silently compare the overlap. Test: Task 7 `test_missing_voxel_fails`.
- `import chroma_lar` and the provenance helper must work on a login node without a GPU (`gpu_name` empty, no crash). Test: Task 4 `test_run_attrs_without_gpu`.
- The generated Slurm scripts for both S3DF and NERSC must point the Triton cache at the job's scratch, not `$HOME`. Test: Task 6 `test_run_script_sets_triton_cache`.

---

### Task 1: pyproject.toml, uv environment, CPU smoke tests

**Files:**
- Create: `pyproject.toml`, `.python-version`, `tests/test_packaging.py`
- Delete: `setup.py`, `requirements.txt`
- Modify: `.gitignore` (add `.venv/`)
- Create (generated): `uv.lock`

**Interfaces:**
- Produces: an environment at `$UV_PROJECT_ENVIRONMENT` with chroma-lar (editable), `trichroma`, photonlib, torch 2.5.1; `uv run pytest` works from the worktree.

- [ ] **Step 1: Write the failing tests**

`tests/test_packaging.py`:

```python
"""chroma-lar installs as a package that runs on TriChroma's Triton backend."""

import os
import subprocess
import sys
from importlib import metadata


def test_distribution_declares_trichroma():
    requires = metadata.distribution("chroma-lar").requires or []
    assert any(r.startswith("trichroma") for r in requires)
    assert not any(r.lower().startswith("pycuda") for r in requires if "extra" not in r)


def test_default_simulation_is_triton():
    env = {k: v for k, v in os.environ.items() if k != "CHROMA_BACKEND"}
    code = "import chroma_lar.geometry; from chroma.sim import Simulation; print(Simulation.__module__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip().splitlines()[-1] == "chroma.triton.simulation"


def test_detector_config_files_are_found():
    import chroma_lar
    from chroma_lar.geometry.config_loader import load_config_from_file

    path = os.path.join(os.path.dirname(chroma_lar.__file__), "config", "detector_config_reflect_reflect3wires.py")
    config = load_config_from_file(path)
    assert config["detector_type"] == "wire"


def test_photocathode_table_is_package_data():
    import chroma_lar

    qe = os.path.join(os.path.dirname(chroma_lar.__file__), "geometry", "custom_optics", "photocathodes",
                      "perfect_pmt_qe.csv")
    assert os.path.exists(qe)
```

- [ ] **Step 2: Run to see them fail**

```bash
cd /sdf/data/neutrino/youngsam/chroma-lar-trichroma
uv run --no-project --with pytest python -m pytest tests/test_packaging.py -q
```
Expected: FAIL (`PackageNotFoundError: chroma-lar`, `No module named 'chroma'`).

- [ ] **Step 3: Write `pyproject.toml`, `.python-version`; delete the old files**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "chroma-lar"
version = "0.1.0"
description = "Liquid-argon detector geometries, optics and lookup-table macros for Chroma (TriChroma)"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "trichroma @ git+https://github.com/youngsm/chroma-lite.git@merge-trichroma",
    "photonlib @ git+https://github.com/CIDeR-ML/photonlib.git",
    "torch==2.5.1",
    "numpy",
    "scipy",
    "h5py",
    "trimesh",
    "PyYAML",
    "pydantic",
    "rich",
    "tqdm",
    "matplotlib",
    "fire",
]

[project.optional-dependencies]
# CUDA Chroma (CHROMA_BACKEND=cuda): compiles PyCUDA, needs the CUDA toolkit.
cuda = ["trichroma[cuda] @ git+https://github.com/youngsm/chroma-lite.git@merge-trichroma"]

[dependency-groups]
dev = ["pytest"]

[tool.uv]
environments = ["sys_platform == 'linux'"]

[tool.setuptools.packages.find]
include = ["chroma_lar*", "data*", "utils*"]

[tool.setuptools.package-data]
chroma_lar = ["config/*.yaml", "geometry/custom_optics/photocathodes/*.csv"]
data = ["*.csv", "*.txt"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.python-version`:

```
3.10
```

```bash
git rm -q setup.py requirements.txt
printf '.venv/\n' >> .gitignore
```

- [ ] **Step 4: Lock, sync, run the tests**

```bash
time uv lock
time uv sync
uv run python -m pytest tests/test_packaging.py -q
```
Expected: `uv lock` resolves `trichroma` to the pushed `merge-trichroma` commit; `uv sync` installs without building PyCUDA; the 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version uv.lock .gitignore tests/test_packaging.py
git commit -q -m "Package chroma-lar with pyproject.toml and uv" -m "Declares TriChroma (Chroma with its Triton backend), photonlib and torch 2.5.1; PyCUDA is the cuda extra. setup.py and requirements.txt are replaced; uv.lock pins the stack."
```

### Task 2: Macros import GPU sources from chroma.triton; fix the 32-bit macro

**Files:**
- Modify: `macros/waveform_map_pyrat.py:14-17`, `macros/waveform_map_pyrat_quantile.py:27-30`
- Modify: `macros/waveform_map_pyrat_32bit.py:172` (stray `-`)
- Create: `tests/test_macros.py`

**Interfaces:**
- Consumes: `chroma.triton.sources.available() -> bool`, `chroma.triton.sources.photon_bombs(nphotons, centres, voxel_size, wavelength, generator)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_macros.py`:

```python
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
```

- [ ] **Step 2: Run to see them fail**

```bash
uv run python -m pytest tests/test_macros.py -q
```
Expected: `test_macro_compiles[waveform_map_pyrat_32bit.py]` FAILS (SyntaxError at line 172) and both `test_gpu_sources_come_from_chroma_triton` FAIL.

- [ ] **Step 3: Fix the macros**

In `macros/waveform_map_pyrat.py` and `macros/waveform_map_pyrat_quantile.py`, replace

```python
try:
    from trichroma import sources  # photons drawn on the GPU (TriChroma's Triton backend)
except ImportError:  # original Chroma: numpy photons
    sources = None
```

with

```python
try:
    from chroma.triton import sources  # photons drawn on the GPU (Triton backend)
except ImportError:  # a Chroma without the Triton engine: numpy photons
    sources = None
```

In `macros/waveform_map_pyrat_32bit.py`, change the line `    )-` (end of the `counts_early` `create_dataset` call) to `    )`.

- [ ] **Step 4: Run the tests**

```bash
uv run python -m pytest tests/test_macros.py -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add macros/waveform_map_pyrat.py macros/waveform_map_pyrat_quantile.py macros/waveform_map_pyrat_32bit.py tests/test_macros.py
git commit -q -m "Macros: GPU photon sources from chroma.triton; fix the 32-bit macro" -m "TriChroma now ships inside chroma as chroma.triton. The 32-bit waveform macro had a stray '-' after a create_dataset call and did not compile."
```

### Task 3: pyrat simulates unweighted photons explicitly

**Files:**
- Modify: `pyrat:140-146`
- Create: `tests/test_pyrat.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pyrat.py`:

```python
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
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run python -m pytest tests/test_pyrat.py -q
```
Expected: FAIL (`assert 'use_weights' in {...}`).

- [ ] **Step 3: Pass `use_weights=False`**

In `pyrat`, replace the `sim.simulate(...)` call with:

```python
        # Lookup tables count photons and take quantiles of single hit times,
        # so photons are never weighted (CHROMA_TRITON=roulette only acts on
        # weighted runs).
        it = sim.simulate(gen, run_daq=db.chroma_daq,
                          photons_per_batch=db.chroma_photons_per_batch,
                          keep_photons_beg=db.chroma_keep_photons_beg,
                          keep_photons_end=db.chroma_keep_photons_end,
                          keep_hits=db.chroma_keep_hits,
                          keep_flat_hits=db.chroma_keep_flat_hits,
                          max_steps=db.chroma_max_steps,
                          use_weights=False)
```

- [ ] **Step 4: Run the test**

```bash
uv run python -m pytest tests/test_pyrat.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyrat tests/test_pyrat.py
git commit -q -m "pyrat: always simulate unweighted photons" -m "Lookup tables count photons and take quantiles of hit times; passing use_weights=False explicitly keeps a changed default from switching LUT runs to weighted mode."
```

### Task 4: `chroma_lar.provenance`

**Files:**
- Create: `chroma_lar/provenance.py`, `tests/test_provenance.py`

**Interfaces:**
- Produces:
  - `source_state(dist_name: str, module_file: str) -> tuple[str, str, int]` — `(version, commit, dirty)`.
  - `run_attrs() -> dict[str, str | int]` — exactly the Global Constraints keys.
  - `write_run_attrs(h5file) -> None` — sets each key on `h5file.attrs`.

- [ ] **Step 1: Write the failing tests**

`tests/test_provenance.py`:

```python
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
    attrs = provenance.run_attrs()
    assert set(attrs) == KEYS
    assert all(isinstance(v, (str, int)) for v in attrs.values())
    assert attrs["backend"] == "triton"
    assert attrs["chroma_triton"] == "legacy"
    assert attrs["use_weights"] == 0
    assert attrs["torch_version"].startswith("2.5.1")


def test_run_attrs_without_gpu(monkeypatch):
    monkeypatch.setattr(provenance, "_gpu_name", lambda: "")
    assert provenance.run_attrs()["gpu_name"] == ""


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
        provenance.write_run_attrs(f)
    with h5py.File(tmp_path / "out.h5", "r") as f:
        assert KEYS <= set(f.attrs)
        assert f.attrs["nphotons_generated"] == 5
        assert f.attrs["use_weights"] == 0
```

- [ ] **Step 2: Run to see them fail**

```bash
uv run python -m pytest tests/test_provenance.py -q
```
Expected: FAIL (`cannot import name 'provenance'`).

- [ ] **Step 3: Implement**

`chroma_lar/provenance.py`:

```python
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


def run_attrs():
    """The attributes describing this run (backend, options, code revisions, GPU)."""
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
        "use_weights": 0,  # pyrat always simulates unweighted photons
        "torch_version": _version("torch"),
        "triton_version": _version("triton"),
        "gpu_name": _gpu_name(),
    }


def write_run_attrs(h5file):
    """Write :func:`run_attrs` into ``h5file``'s root attributes."""
    for key, value in run_attrs().items():
        h5file.attrs[key] = value
```

- [ ] **Step 4: Run the tests**

```bash
uv run python -m pytest tests/test_provenance.py -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add chroma_lar/provenance.py tests/test_provenance.py
git commit -q -m "Add chroma_lar.provenance: backend, options and code revisions of a run" -m "Commits come from a git install's direct_url.json or from the checkout an editable install lives in; site-packages copies are never matched to an enclosing repository."
```

### Task 5: Waveform-map macros write provenance

**Files:**
- Modify: `macros/waveform_map_pyrat.py` (after `db.file = h5py.File(...)` in `__simulation_start__`)
- Modify: `macros/waveform_map_pyrat_32bit.py` (same place)
- Modify: `macros/waveform_map_pyrat_quantile.py` (after the `# Metadata for downstream reconstruction.` attributes)
- Modify: `tests/test_macros.py`

**Interfaces:**
- Consumes: `chroma_lar.provenance.write_run_attrs(h5file)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_macros.py`:

```python
WAVEFORM_MACROS = ["waveform_map_pyrat.py", "waveform_map_pyrat_32bit.py", "waveform_map_pyrat_quantile.py"]


@pytest.mark.parametrize("name", WAVEFORM_MACROS)
def test_waveform_macro_writes_provenance(name):
    tree = ast.parse((ROOT / "macros" / name).read_text())
    start = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "__simulation_start__")
    calls = [node for node in ast.walk(start)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "write_run_attrs"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "db.file"
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run python -m pytest tests/test_macros.py -q -k provenance
```
Expected: 3 FAIL (`StopIteration` is not raised; `assert 0 == 1`).

- [ ] **Step 3: Add the calls**

In each of the three macros, add the import next to the other `chroma_lar` import:

```python
from chroma_lar.provenance import write_run_attrs
```

In `waveform_map_pyrat.py` and `waveform_map_pyrat_32bit.py`, directly after `db.file = h5py.File(db.output_filename, "w")`:

```python
    write_run_attrs(db.file)  # backend, CHROMA_TRITON options, code revisions
```

In `waveform_map_pyrat_quantile.py`, directly after `db.file.attrs["audit_every_n_voxels"] = int(db.audit_every_n_voxels)`:

```python
    write_run_attrs(db.file)  # backend, CHROMA_TRITON options, code revisions
```

- [ ] **Step 4: Run the tests**

```bash
uv run python -m pytest tests/test_macros.py -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add macros/waveform_map_pyrat.py macros/waveform_map_pyrat_32bit.py macros/waveform_map_pyrat_quantile.py tests/test_macros.py
git commit -q -m "Waveform-map macros record their run's provenance" -m "Each output file's root attributes say which backend and CHROMA_TRITON options made it and which TriChroma and chroma-lar revisions ran."
```

### Task 6: Job scripts keep the Triton cache in scratch

**Files:**
- Modify: `jobs/submit_waveform_jobs.py:110` (local `--run-job` command) and `:249` (`run.sh` template)
- Create: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

`tests/test_jobs.py`:

```python
"""The generated run scripts send Triton's kernel cache to the job's scratch."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CONFIG = '''
config = {{
    "detector_x_range": (-2310, 0), "detector_y_range": (-2160, 2160), "detector_z_range": (-2160, 2160),
    "voxel_size": 30, "nphotons": 1000, "detector_config": "detector_config_reflect_reflect3wires",
    "time_per_voxel": 1, "max_job_time": 3600, "slurm_max_job_time_buffer": 600, "site": "{site}",
    "macro": "waveform_map_pyrat_quantile.py",
}}
site = dict({site}=dict(work_dir="$TMPDIR", container_cmd="true", output_dir="{out}",
                        slurm=dict(output="{out}/logs/%A_%a.log", error="{out}/logs/%A_%a.log")))
'''


def test_run_script_sets_triton_cache(tmp_path):
    for site in ("slac", "perlmutter"):
        out = tmp_path / site
        config = tmp_path / ("config_%s.py" % site)
        config.write_text(CONFIG.format(site=site, out=out))
        subprocess.run([sys.executable, str(ROOT / "jobs" / "submit_waveform_jobs.py"), "--dry-run",
                        "--config", str(config)], check=True, capture_output=True, cwd=ROOT / "jobs")
        run = (out / "run.sh").read_text()
        assert "export TRITON_CACHE_DIR=$PWD/tmp/triton" in run
        assert "export PYCUDA_CACHE_DIR=$PWD/tmp" in run
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run python -m pytest tests/test_jobs.py -q
```
Expected: FAIL on the `TRITON_CACHE_DIR` assertion.

- [ ] **Step 3: Edit the templates**

In `jobs/submit_waveform_jobs.py`, in the `run_script` template, change

```bash
mkdir -p tmp
export PYCUDA_CACHE_DIR=$PWD/tmp
```

to

```bash
mkdir -p tmp/triton
export PYCUDA_CACHE_DIR=$PWD/tmp
export TRITON_CACHE_DIR=$PWD/tmp/triton
```

and in the `--run-job` command, change `PYCUDA_CACHE_DIR=/lscratch singularity exec` to `PYCUDA_CACHE_DIR=/lscratch TRITON_CACHE_DIR=/lscratch/$USER/triton-cache singularity exec`.

- [ ] **Step 4: Run the test**

```bash
uv run python -m pytest tests/test_jobs.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jobs/submit_waveform_jobs.py tests/test_jobs.py
git commit -q -m "Job scripts: Triton's kernel cache goes to the job's scratch" -m "TRITON_CACHE_DIR sits next to PYCUDA_CACHE_DIR, so compiled kernels never land in \$HOME."
```

### Task 7: `chroma_lar.validation` — compare two waveform-map runs

**Files:**
- Create: `chroma_lar/validation.py`, `tests/test_validation.py`

**Interfaces:**
- Consumes: `chroma.triton.utils.two_proportion_z(k1, n1, k2, n2) -> float`, `chroma.triton.utils.ks_test(a, b, alpha) -> dict | None` (`{"distance", "bound", "pass"}`).
- Produces:
  - `load_hits(path) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], int, int]` — `({voxel_id: (times, channels)}, nphotons_per_voxel, num_channels)` from the audit datasets.
  - `compare(path_a, path_b, max_sigma=5.0, alpha=1e-6, min_hits=25) -> dict` with keys `voxels`, `missing`, `tests`, `z_max`, `z_mean`, `z_std`, `worst`, `ks_failures`, `yield_ratio`, `passed`.
  - CLI: `python -m chroma_lar.validation A.h5 B.h5 [--json OUT] [--max-sigma 5]`; exit 0 when `passed`, else 1.

- [ ] **Step 1: Write the failing tests**

`tests/test_validation.py`:

```python
"""Statistical comparison of two waveform-map runs (audit hits for every voxel)."""

import h5py
import numpy as np
import pytest

from chroma_lar import validation

NPHOTONS = 1_000_000
NCHAN = 20
VOXELS = list(range(100, 108))


def _write(path, seed, yield_scale=1.0, time_shift=0.0, voxels=VOXELS):
    rng = np.random.default_rng(seed)
    lam = np.linspace(200, 2000, NCHAN) * yield_scale
    t_all, ch_all, offsets, nhits = [], [], [], []
    offset = 0
    for _ in voxels:
        counts = rng.poisson(lam)
        ch = np.repeat(np.arange(NCHAN), counts).astype(np.uint16)
        t = (rng.exponential(10.0, ch.size) + ch * 0.5 + time_shift).astype(np.float32)
        t_all.append(t); ch_all.append(ch); offsets.append(offset); nhits.append(ch.size)
        offset += ch.size
    with h5py.File(path, "w") as f:
        f.attrs["nphotons_generated"] = NPHOTONS
        f.attrs["num_output_channels"] = NCHAN
        f["audit_t"] = np.concatenate(t_all)
        f["audit_channel"] = np.concatenate(ch_all)
        f["audit_voxel_id"] = np.asarray(voxels, dtype=np.uint32)
        f["audit_offset"] = np.asarray(offsets, dtype=np.uint64)
        f["audit_nhits"] = np.asarray(nhits, dtype=np.uint32)
    return path


def test_load_hits(tmp_path):
    hits, nphotons, nchan = validation.load_hits(_write(tmp_path / "a.h5", 1))
    assert (nphotons, nchan) == (NPHOTONS, NCHAN)
    assert sorted(hits) == VOXELS
    t, ch = hits[100]
    assert t.shape == ch.shape


def test_same_distribution_passes(tmp_path):
    result = validation.compare(_write(tmp_path / "a.h5", 1), _write(tmp_path / "b.h5", 2))
    assert result["passed"], result
    assert result["tests"] == len(VOXELS) * NCHAN
    assert abs(result["z_mean"]) < 0.4 and 0.7 < result["z_std"] < 1.3
    # each voxel has ~22k hits per run: the ratio's standard error is ~1%
    assert all(abs(r - 1) < 0.05 for r in result["yield_ratio"].values())


def test_more_light_fails_and_reports_ratio(tmp_path):
    result = validation.compare(_write(tmp_path / "a.h5", 1), _write(tmp_path / "b.h5", 2, yield_scale=1.18))
    assert not result["passed"]
    assert all(abs(r - 1.18) < 0.06 for r in result["yield_ratio"].values())
    assert result["z_mean"] > 1  # every channel is brighter


def test_later_times_fail_ks(tmp_path):
    result = validation.compare(_write(tmp_path / "a.h5", 1), _write(tmp_path / "b.h5", 2, time_shift=3.0))
    assert not result["passed"]
    assert result["ks_failures"]


def test_missing_voxel_fails(tmp_path):
    result = validation.compare(_write(tmp_path / "a.h5", 1), _write(tmp_path / "b.h5", 2, voxels=VOXELS[:-1]))
    assert not result["passed"]
    assert result["missing"] == [VOXELS[-1]]


def test_channel_count_mismatch_raises(tmp_path):
    a = _write(tmp_path / "a.h5", 1)
    b = _write(tmp_path / "b.h5", 2)
    with h5py.File(b, "a") as f:
        f.attrs["num_output_channels"] = NCHAN + 1
    with pytest.raises(ValueError):
        validation.compare(a, b)


def test_cli_exit_code(tmp_path):
    a = _write(tmp_path / "a.h5", 1)
    assert validation.main([str(a), str(_write(tmp_path / "b.h5", 2)), "--json", str(tmp_path / "r.json")]) == 0
    assert validation.main([str(a), str(_write(tmp_path / "c.h5", 3, yield_scale=1.18))]) == 1
    assert (tmp_path / "r.json").exists()
```

- [ ] **Step 2: Run to see them fail**

```bash
uv run python -m pytest tests/test_validation.py -q
```
Expected: FAIL (`cannot import name 'validation'`).

- [ ] **Step 3: Implement**

`chroma_lar/validation.py`:

```python
"""Statistical comparison of two waveform-map runs of the same voxels.

Both files come from ``macros/waveform_map_pyrat_quantile.py`` with raw hits
kept for every voxel (``audit_every_n_voxels=1``). For every voxel and channel
with at least ``min_hits`` hits in the two runs together, the yields are
compared with a two-proportion z test and the hit times with a two-sample KS
test against its DKW bound at ``alpha``. The runs agree (``passed``) when no
|z| exceeds ``max_sigma``, the z values have mean near 0 and spread near 1
(within five standard errors), no KS test fails, and both runs have the same
voxels.

Usage: python -m chroma_lar.validation A.h5 B.h5 [--json OUT] [--max-sigma 5]
"""

import argparse
import json
import sys

import h5py
import numpy as np

from chroma.triton.utils import ks_test, two_proportion_z


def load_hits(path):
    """``({voxel_id: (times, channels)}, nphotons per voxel, channels)`` of a run's raw hits."""
    with h5py.File(path, "r") as f:
        nphotons = int(f.attrs["nphotons_generated"])
        nchan = int(f.attrs["num_output_channels"])
        voxel_ids = f["audit_voxel_id"][:]
        offsets = f["audit_offset"][:]
        nhits = f["audit_nhits"][:]
        times = f["audit_t"][:]
        channels = f["audit_channel"][:]
    hits = {int(v): (times[o:o + n], channels[o:o + n]) for v, o, n in zip(voxel_ids, offsets, nhits)}
    return hits, nphotons, nchan


def compare(path_a, path_b, max_sigma=5.0, alpha=1e-6, min_hits=25):
    """Compare run B against run A; see the module docstring."""
    hits_a, n_a, nchan = load_hits(path_a)
    hits_b, n_b, nchan_b = load_hits(path_b)
    if nchan != nchan_b:
        raise ValueError("the runs have %d and %d output channels" % (nchan, nchan_b))
    voxels = sorted(set(hits_a) & set(hits_b))
    z_values, ks_failures, ratios, worst = [], [], {}, None
    for v in voxels:
        times_a, chan_a = hits_a[v]
        times_b, chan_b = hits_b[v]
        k_a = np.bincount(chan_a, minlength=nchan)
        k_b = np.bincount(chan_b, minlength=nchan)
        ratios[v] = float((k_b.sum() / n_b) / (k_a.sum() / n_a)) if k_a.sum() else float("nan")
        for c in np.nonzero(k_a + k_b >= min_hits)[0]:
            z = two_proportion_z(int(k_a[c]), n_a, int(k_b[c]), n_b)
            z_values.append(z)
            if worst is None or abs(z) > abs(worst["z"]):
                worst = {"z": float(z), "voxel": v, "channel": int(c)}
            ks = ks_test(times_a[chan_a == c], times_b[chan_b == c], alpha=alpha)
            if ks is not None and not ks["pass"]:
                ks_failures.append({"voxel": v, "channel": int(c), "distance": ks["distance"],
                                    "bound": ks["bound"]})
    z = np.asarray(z_values, dtype=np.float64)
    result = {
        "voxels": len(voxels),
        "missing": sorted(set(hits_a) ^ set(hits_b)),
        "tests": int(z.size),
        "z_max": float(np.abs(z).max()) if z.size else 0.0,
        "z_mean": float(z.mean()) if z.size else 0.0,
        "z_std": float(z.std()) if z.size else 0.0,
        "worst": worst,
        "ks_failures": ks_failures,
        "yield_ratio": {str(v): r for v, r in ratios.items()},
    }
    n = max(z.size, 1)
    result["passed"] = bool(
        z.size
        and result["z_max"] <= max_sigma
        and abs(result["z_mean"]) <= 5 / np.sqrt(n)
        and abs(result["z_std"] - 1) <= 5 / np.sqrt(2 * n)
        and not ks_failures
        and not result["missing"]
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--json", help="write the full result here")
    parser.add_argument("--max-sigma", type=float, default=5.0)
    args = parser.parse_args(argv)
    result = compare(args.a, args.b, max_sigma=args.max_sigma)
    summary = {k: result[k] for k in ("passed", "voxels", "missing", "tests", "z_max", "z_mean", "z_std", "worst")}
    summary["ks_failures"] = len(result["ks_failures"])
    ratios = list(result["yield_ratio"].values())
    summary["yield_ratio_mean"] = float(np.nanmean(ratios)) if ratios else float("nan")
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

```bash
uv run python -m pytest tests/test_validation.py -q
```
Expected: all pass. If `test_same_distribution_passes` fails only on the `z_std` window, print `result` and check the synthetic data, not the thresholds: the spread of independent Poisson z values is 1.

- [ ] **Step 5: Commit**

```bash
git add chroma_lar/validation.py tests/test_validation.py
git commit -q -m "Add chroma_lar.validation: compare two waveform-map runs statistically" -m "Per voxel and channel: two-proportion z of the yields and KS of the hit times, from the macro's audit hits; used to validate TriChroma against CUDA Chroma."
```

### Task 8: README

**Files:**
- Modify: `README.md` (sections `### Installing chroma-lite` through the `env.sh` note; new section before `## Geometries`)
- Create: `tests/test_readme.py`

- [ ] **Step 1: Write the failing test**

`tests/test_readme.py`:

```python
"""The README installs with uv and states the LUT rules."""

import pathlib

README = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()


def test_install_is_uv():
    assert "uv sync" in README
    assert "pip install -e git+https://github.com/youngsm/chroma-lite" not in README
    assert "source chroma-lar/env.sh" not in README


def test_lut_rules_are_documented():
    for phrase in ("use_weights=False", "CHROMA_TRITON=legacy", "CHROMA_BACKEND=cuda", "18%",
                   "chroma_lar/provenance.py"):
        assert phrase in README
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run python -m pytest tests/test_readme.py -q
```
Expected: FAIL.

- [ ] **Step 3: Edit the README**

Replace everything from `### Installing chroma-lite` up to (not including) `## Geometries` with:

````markdown
### Installing

chroma-lar runs on [TriChroma](https://github.com/youngsm/trichroma): Chroma
with a Triton GPU backend, installed together with chroma-lar. Nothing is
compiled and PyCUDA is not needed. Simulations need Linux and an NVIDIA GPU;
building geometry works without one.

```bash
git clone https://github.com/youngsm/chroma-lar.git
cd chroma-lar
uv sync                       # .venv with chroma-lar, TriChroma, PyTorch, Triton
source .venv/bin/activate
source chroma_lar_env.sh      # puts pyrat on PATH
```

`uv sync --extra cuda` also installs PyCUDA for the original CUDA backend
(`CHROMA_BACKEND=cuda`); it compiles PyCUDA and needs the CUDA toolkit. Keep
`UV_CACHE_DIR` and `TRITON_CACHE_DIR` (compiled kernels) on a large disk; on
S3DF, under `/sdf/data`.

### Backends and lookup tables

`chroma.sim.Simulation` runs on TriChroma's Triton engine by default, in its
production mode, which fixes several CUDA Chroma bugs. On this repository's
wire TPC the wire-plane fix alone raises the detected light by about 18%, so
new lookup tables are not comparable to ones made with CUDA Chroma (b04, b05).
`CHROMA_TRITON=legacy` restores CUDA Chroma's behaviour; `CHROMA_BACKEND=cuda`
runs CUDA Chroma itself.

Lookup tables are made with unweighted photons: `yield_counts` are photon
counts and `quantiles` are quantiles of single hit times. `pyrat` always calls
`simulate(..., use_weights=False)`; `CHROMA_TRITON=roulette=<w>` only acts on
weighted runs and has no effect here.

The waveform-map macros record how each file was made in its root attributes
(`backend`, `chroma_triton`, `trichroma_commit`, `chroma_lar_commit`, ...; see
`chroma_lar/provenance.py`). `python -m chroma_lar.validation A.h5 B.h5`
compares two runs of the same voxels statistically.

````

- [ ] **Step 4: Run the test**

```bash
uv run python -m pytest tests/test_readme.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme.py
git commit -q -m "README: install with uv; backends, unweighted LUTs and provenance"
```

### Task 9: Container recipes

**Files:**
- Modify: `installation/chroma3.lar-plib/Singularity`, `installation/chroma3.lar-plib/Dockerfile`

**Interfaces:**
- Produces: images with Python at `/opt/conda/bin/python`, chroma-lar editable at `/workspace/chroma-lar` (with `.git`, so provenance reports its commit), `pyrat` on `PATH`, no PyCUDA. Build argument `CHROMA_LAR_REF` (default `main`).

- [ ] **Step 1: Rewrite the Singularity recipe**

`installation/chroma3.lar-plib/Singularity`:

```
Bootstrap: docker
From: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Build: apptainer build --fakeroot [--build-arg CHROMA_LAR_REF=<branch>] chroma-lar.sif Singularity
%arguments
    CHROMA_LAR_REF=main

%post
    apt-get update && apt-get install -y \
        git \
        wget \
        build-essential \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        freeglut3-dev \
        libglew-dev \
        libglfw3-dev \
        libegl1-mesa-dev \
        libgbm-dev \
        libgles2-mesa-dev \
        mesa-utils \
        libvulkan1 \
        mesa-vulkan-drivers \
        libx11-dev \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev  \
        mesa-common-dev \
        xorg-dev \
        && rm -rf /var/lib/apt/lists/*

    /opt/conda/bin/python -m pip install --no-cache-dir uv==0.11.28
    export UV_CACHE_DIR=/tmp/uv-cache
    mkdir -p /workspace && cd /workspace
    git clone --branch {{ CHROMA_LAR_REF }} https://github.com/youngsm/chroma-lar
    # chroma-lar, TriChroma (Chroma + its Triton backend) and photonlib; no PyCUDA
    uv pip install --python /opt/conda/bin/python -e /workspace/chroma-lar
    # analysis extras
    uv pip install --python /opt/conda/bin/python scikit-learn gdown \
        git+https://github.com/CIDeR-ML/siren-lartpc.git \
        git+https://github.com/CIDeR-ML/sirentv.git
    rm -rf /tmp/uv-cache

%environment
    export PYRAT_ROOT=/workspace/chroma-lar
    export PATH=/workspace/chroma-lar:$PATH
    #For jupyter
    export PORT=8888

%startscript
    /bin/bash -l
```

- [ ] **Step 2: Rewrite the Dockerfile**

`installation/chroma3.lar-plib/Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ARG CHROMA_LAR_REF=main

RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/python -m pip install --no-cache-dir uv==0.11.28
WORKDIR /workspace/
ADD https://api.github.com/repos/youngsm/chroma-lar/git/refs/heads/${CHROMA_LAR_REF} /opt/chroma_lar_version.json
RUN git clone --branch ${CHROMA_LAR_REF} https://github.com/youngsm/chroma-lar
# chroma-lar, TriChroma (Chroma + its Triton backend) and photonlib; no PyCUDA
RUN uv pip install --no-cache --python /opt/conda/bin/python -e /workspace/chroma-lar
# analysis extras
RUN uv pip install --no-cache --python /opt/conda/bin/python scikit-learn gdown \
    git+https://github.com/CIDeR-ML/siren-lartpc.git \
    git+https://github.com/CIDeR-ML/sirentv.git
ENV PYRAT_ROOT=/workspace/chroma-lar
ENV PATH=/workspace/chroma-lar:$PATH

#For jupyter
EXPOSE 8888

CMD ["/bin/bash","-l"]
```

- [ ] **Step 3: Check the old chroma-lite steps are gone**

```bash
grep -n "chroma-lite\|camera.py\|VIRTUAL_ENV\|cupy\|pycuda" installation/chroma3.lar-plib/Singularity installation/chroma3.lar-plib/Dockerfile
```
Expected: no output. (The `camera.py` line-741 fix patched chroma-lite's viewer, which needs the `cuda` and `viz` extras and is not in these images.)

- [ ] **Step 4: Commit**

```bash
git add installation/chroma3.lar-plib/Singularity installation/chroma3.lar-plib/Dockerfile
git commit -q -m "Containers: install chroma-lar and TriChroma with uv, no PyCUDA" -m "Both recipes start from pytorch 2.5.1/CUDA 12.4 (Singularity was on 11.8), drop the hand clone of chroma-lite, and take the chroma-lar branch as a build argument."
```

(The build itself is Task 11.)

### Task 10: Validation scripts

**Files:**
- Create: `scripts/validation/validation_voxels.py`, `scripts/validation/compare_backends.sh`

**Interfaces:**
- Consumes: `python -m chroma_lar.validation` (Task 7); `python -m chroma.triton.tape.verify` and `benchmarks/simulate_throughput.py` from the TriChroma checkout `/sdf/data/neutrino/youngsam/trichroma-merge`; the image and CUDA environment built in Task 11.
- Produces: `compare_backends.sh <label> <output dir>`, run on a GPU node; writes `summary.txt`, `bitwise.txt`, `compare_legacy.txt`, `compare_production.txt`, `provenance.txt`, `throughput_*.txt` under `<output dir>/<label>/`.

These scripts are committed before the image is built (Task 11), so the image contains them.

- [ ] **Step 1: The voxel list**

`scripts/validation/validation_voxels.py`:

```python
"""Write the 8 validation voxels (cathode, anode, walls, corners) of the 3 cm
reflect3wires LUT grid to the .npy file given as the only argument."""

import sys

import numpy as np
import torch
from photonlib.meta import VoxelMeta

RANGES = ((-2310, 0), (-2160, 2160), (-2160, 2160))
VOXEL_SIZE = 30
POINTS = [
    (-60, 0, 0), (-1155, 0, 0), (-2250, 0, 0),              # cathode, middle, anode
    (-1155, 2100, 0), (-1155, -2100, 0), (-1155, 0, 2100),  # walls
    (-2250, 2100, 2100), (-60, -2100, -2100),               # corners
]

shape = tuple((hi - lo) // VOXEL_SIZE for lo, hi in RANGES)
meta = VoxelMeta(shape=shape, ranges=RANGES)
ids = meta.coord_to_voxel(torch.tensor(POINTS, dtype=torch.float32)).numpy().astype(np.int32)
assert len(set(ids.tolist())) == len(POINTS), ids
np.save(sys.argv[1], ids)
print(ids)
```

Check it on the CPU:

```bash
uv run python scripts/validation/validation_voxels.py /sdf/data/neutrino/youngsam/tmp/voxels.npy
```
Expected: 8 distinct voxel IDs printed.

- [ ] **Step 2: The validation driver**

`scripts/validation/compare_backends.sh`:

```bash
#!/bin/bash
# GPU validation of TriChroma against CUDA Chroma (run on one GPU node).
# Usage: compare_backends.sh <label> <output dir>
set -u
LABEL=$1
OUT=$2/$LABEL
SIF=/sdf/data/neutrino/youngsam/containers/chroma-lar-trichroma.sif
CUDAPY=/sdf/data/neutrino/youngsam/envs/chroma-lar-cuda/bin/python
TRI=/sdf/data/neutrino/youngsam/trichroma-merge
LAR=/workspace/chroma-lar
NPHOTONS=${NPHOTONS:-2000000}
mkdir -p "$OUT"
export TRITON_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/triton
export PYCUDA_CACHE_DIR=$OUT/pycuda-cache
run() { apptainer exec --nv -B /sdf "$SIF" bash -c "cd $OUT && $*"; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"

run "python $LAR/scripts/validation/validation_voxels.py $OUT/voxels.npy"

PYRAT_ARGS="-s detector_config detector_config_reflect_reflect3wires \
  -es voxel_ranges '((-2310,0),(-2160,2160),(-2160,2160))' -es voxel_size 30 -es nphotons $NPHOTONS \
  -s voxel_id_file $OUT/voxels.npy -es voxel_index_start 0 -es batch_size 8 -es audit_every_n_voxels 1"

# 1. Bitwise: CUDA Chroma recorded, replayed by chroma.triton, compared
run "$CUDAPY -m chroma.triton.tape.verify all --fixture reflect3wires --work $OUT/tapes" \
  > "$OUT/bitwise.txt" 2>&1; echo "bitwise exit $?" | tee "$OUT/summary.txt"

# 2. LUT runs: CUDA Chroma, TriChroma legacy, TriChroma production
run "CHROMA_BACKEND=cuda $CUDAPY $LAR/pyrat $LAR/macros/waveform_map_pyrat_quantile.py $PYRAT_ARGS -s output_filename $OUT/cuda.h5" \
  > "$OUT/run_cuda.txt" 2>&1; echo "run_cuda exit $?" | tee -a "$OUT/summary.txt"
run "CHROMA_TRITON=legacy python $LAR/pyrat $LAR/macros/waveform_map_pyrat_quantile.py $PYRAT_ARGS -s output_filename $OUT/legacy.h5" \
  > "$OUT/run_legacy.txt" 2>&1; echo "run_legacy exit $?" | tee -a "$OUT/summary.txt"
run "python $LAR/pyrat $LAR/macros/waveform_map_pyrat_quantile.py $PYRAT_ARGS -s output_filename $OUT/production.h5" \
  > "$OUT/run_production.txt" 2>&1; echo "run_production exit $?" | tee -a "$OUT/summary.txt"

# 3. Statistics
run "python -m chroma_lar.validation $OUT/cuda.h5 $OUT/legacy.h5 --json $OUT/legacy.json" \
  > "$OUT/compare_legacy.txt" 2>&1; echo "compare_legacy exit $?" | tee -a "$OUT/summary.txt"
run "python -m chroma_lar.validation $OUT/cuda.h5 $OUT/production.h5 --json $OUT/production.json" \
  > "$OUT/compare_production.txt" 2>&1; echo "compare_production exit $? (informational)" | tee -a "$OUT/summary.txt"

# 4. Provenance
run "python -c \"
import h5py
for name in ('cuda', 'legacy', 'production'):
    with h5py.File('$OUT/' + name + '.h5', 'r') as f:
        print(name, {k: f.attrs[k] for k in ('backend', 'chroma_triton', 'trichroma_commit', 'chroma_lar_commit', 'use_weights', 'gpu_name')})
\"" > "$OUT/provenance.txt" 2>&1

# 5. Throughput (LUT-style, unweighted, 450 nm)
run "CHROMA_BACKEND=cuda $CUDAPY $TRI/benchmarks/simulate_throughput.py 10000000 --wavelength 450" \
  > "$OUT/throughput_cuda.txt" 2>&1
run "python $TRI/benchmarks/simulate_throughput.py 30000000 --wavelength 450" \
  > "$OUT/throughput_triton.txt" 2>&1
cat "$OUT/summary.txt"
```

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/validation/compare_backends.sh
git add scripts/validation
git commit -q -m "Validation scripts: TriChroma against CUDA Chroma on a GPU node" -m "Bitwise tape replay, LUT runs on both backends compared voxel by voxel, provenance and throughput."
```

### Task 11: Cold install timing, push, image build

**Files:**
- Create: `/sdf/data/neutrino/youngsam/containers/chroma-lar-trichroma.sif`
- Create: `/sdf/data/neutrino/youngsam/envs/chroma-lar-cuda` (venv inside the image, with PyCUDA)

- [ ] **Step 1: Full CPU test suite**

```bash
uv run python -m pytest -q
```
Expected: all pass.

- [ ] **Step 2: Cold install, timed**

```bash
rm -rf /sdf/data/neutrino/youngsam/.cache/uv-cold /sdf/data/neutrino/youngsam/envs/chroma-lar-cold
time UV_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/uv-cold \
  UV_PROJECT_ENVIRONMENT=/sdf/data/neutrino/youngsam/envs/chroma-lar-cold uv sync --frozen
rm -rf /sdf/data/neutrino/youngsam/.cache/uv-cold /sdf/data/neutrino/youngsam/envs/chroma-lar-cold
```
Expected: completes without compiling; record the wall time for the report.

- [ ] **Step 3: Push the branch (pre-approved)**

The image clones the branch from GitHub. Push only if Steps 1-2 passed:

```bash
git push -u origin trichroma-default
```

- [ ] **Step 4: Build the image**

```bash
mkdir -p $APPTAINER_TMPDIR /sdf/data/neutrino/youngsam/containers
time apptainer build --fakeroot --build-arg CHROMA_LAR_REF=trichroma-default \
  /sdf/data/neutrino/youngsam/containers/chroma-lar-trichroma.sif installation/chroma3.lar-plib/Singularity
apptainer exec /sdf/data/neutrino/youngsam/containers/chroma-lar-trichroma.sif \
  python -c "import chroma.sim, chroma_lar, importlib.util as u; print(chroma.sim.Simulation.__module__, u.find_spec('pycuda') is None)"
```
Expected: `chroma.triton.simulation True`. If `--fakeroot` fails (no subuid mapping), report the error to the user; do not try other privilege workarounds.

- [ ] **Step 5: The CUDA reference environment (PyCUDA inside the image)**

```bash
apptainer exec -B /sdf /sdf/data/neutrino/youngsam/containers/chroma-lar-trichroma.sif bash -c '
  export UV_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/uv
  /opt/conda/bin/uv venv --system-site-packages --python /opt/conda/bin/python /sdf/data/neutrino/youngsam/envs/chroma-lar-cuda
  /opt/conda/bin/uv pip install --python /sdf/data/neutrino/youngsam/envs/chroma-lar-cuda/bin/python pycuda "pytools==2022.1.2"
  CHROMA_BACKEND=cuda /sdf/data/neutrino/youngsam/envs/chroma-lar-cuda/bin/python -c "import chroma.sim as s; print(s.Simulation.__module__)"'
```
Expected: `chroma.sim_cuda`. (These are exactly the `cuda` extra's packages; chroma-lar and TriChroma come from the image.)

### Task 12: GPU validation on ampere and turing

**Files:**
- Create: `docs/validation/2026-10-14-trichroma-validation.md`

**Interfaces:**
- Consumes: `scripts/validation/compare_backends.sh` (Task 10), the image and CUDA environment (Task 11).

- [ ] **Step 1: Run on both GPUs**

```bash
OUTDIR=/sdf/data/neutrino/youngsam/validation/trichroma-default
for p in ampere turing; do
  srun --account=neutrino:cider-nu --partition=$p --gpus=1 --cpus-per-gpu=4 --mem=48G --time=3:00:00 \
    /sdf/data/neutrino/youngsam/chroma-lar-trichroma/scripts/validation/compare_backends.sh $p $OUTDIR
done
```

- [ ] **Step 2: Check against the spec's criteria**

```bash
OUTDIR=/sdf/data/neutrino/youngsam/validation/trichroma-default
for p in ampere turing; do echo "== $p"; cat $OUTDIR/$p/summary.txt; cat $OUTDIR/$p/compare_legacy.txt $OUTDIR/$p/compare_production.txt; tail -5 $OUTDIR/$p/bitwise.txt; cat $OUTDIR/$p/provenance.txt; tail -3 $OUTDIR/$p/throughput_*.txt; done
```

Pass, on each GPU:
- `bitwise exit 0` and the verify report says every output is identical.
- `run_*` exit 0.
- `compare_legacy`: `passed: true` (no |z| > 5, z mean and spread within five standard errors of 0 and 1, no KS failure, no missing voxel).
- `compare_production`: reported, not judged; `yield_ratio_mean` near the guide's +18% (about 1.18).
- `provenance.txt`: `backend` is `cuda` for cuda.h5 and `triton` for the others; `chroma_triton` is `legacy` only for legacy.h5; `use_weights` 0; commits non-empty.

If a pass criterion fails, stop and report it with the files; do not change thresholds.

- [ ] **Step 3: Write and commit the report**

`docs/validation/2026-10-14-trichroma-validation.md`: a short table per GPU (GPU name and driver; bitwise result; legacy z_max, z_mean, z_std, KS failures; production yield ratio and z summary; photons/s for CUDA and TriChroma; cold `uv sync` time from Task 11 Step 2; image build time), the exact commits (`trichroma_commit`, `chroma_lar_commit` from the provenance), and the output directory path.

```bash
git add docs/validation/2026-10-14-trichroma-validation.md
git commit -q -m "Validation report: TriChroma against CUDA Chroma on A100 and RTX 2080 Ti"
```

### Task 13: Final dependency URL (after the GitHub rename)

Run only after the merge plan's Task 10 (merge to master, repository renames) and the user's yes.

- [ ] **Step 1: Point at the renamed repository**

In `pyproject.toml`, replace both occurrences of `git+https://github.com/youngsm/chroma-lite.git@merge-trichroma` with `git+https://github.com/youngsm/trichroma.git`.

- [ ] **Step 2: Re-lock and test**

```bash
uv lock
uv sync
uv run python -m pytest -q
grep -n "chroma-lite" pyproject.toml uv.lock README.md installation/chroma3.lar-plib/*
```
Expected: tests pass; no `chroma-lite` URL left.

- [ ] **Step 3: Commit and push (after yes)**

```bash
git add pyproject.toml uv.lock
git commit -q -m "Depend on youngsm/trichroma (the renamed chroma-lite)"
git push
```
