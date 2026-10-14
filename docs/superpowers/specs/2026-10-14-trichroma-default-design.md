# TriChroma as chroma-lar's default backend

Date: 2026-10-14. Branch: `trichroma-default`, from `origin/trichroma-fast-waveform`.

## Goal

chroma-lar runs its simulations on TriChroma (Triton) by default. Installing it
takes a few minutes with uv and compiles nothing; PyCUDA is an optional extra.
The CUDA backend stays available, and LUT generation stays unweighted.

## What exists

- chroma-lite `master` picks the implementation behind `chroma.sim.Simulation`
  from `CHROMA_BACKEND` (`cuda` or `triton`). Unset, it is `cuda` when PyCUDA is
  installed and `triton` otherwise. PyCUDA is chroma-lite's `[cuda]` extra, and
  its loader builds no Chroma BVH on the Triton backend.
- TriChroma `main` provides `trichroma.simulation.Simulation` with the same
  constructor and `simulate` signature as CUDA Chroma, plus
  `trichroma.sources` (photons drawn on the GPU).
- chroma-lar `origin/trichroma-fast-waveform` makes the waveform macros draw
  photons with `trichroma.sources` when it is importable.
- chroma-lar has a `setup.py` with no dependencies, a loose `requirements.txt`,
  and container recipes that clone chroma-lite by hand.

## Packaging

`pyproject.toml` (setuptools backend) replaces `setup.py` and
`requirements.txt`. It finds the same packages `find_packages()` found
(`chroma_lar`, `data`, `utils`). `requires-python = ">=3.10"`.

Dependencies, as PEP 508 direct references so both pip and uv resolve them:

| Requirement | Why |
|---|---|
| `chroma @ git+https://github.com/youngsm/chroma-lite.git` | Chroma's API (chroma-lite's distribution name is `Chroma`) |
| `trichroma @ git+https://github.com/youngsm/trichroma.git` | Triton backend |
| `photonlib @ git+https://github.com/CIDeR-ML/photonlib.git` | `VoxelMeta` in the macros |
| `torch==2.5.1` | TriChroma's validated stack; brings `triton==3.1.0`; PyPI wheels are cu124 |
| numpy, scipy, h5py, trimesh, PyYAML, pydantic, rich, tqdm, matplotlib, fire | used by the package, macros and job scripts |

Extra `cuda = ["chroma[cuda] @ git+https://github.com/youngsm/chroma-lite.git"]`
adds PyCUDA; nothing else compiles.

`uv.lock` is committed. Install: `uv sync` (add `--extra cuda` for PyCUDA).
`chroma_lar_env.sh` and `pyrat` keep working as before.

## Default backend

`chroma_lar/__init__.py` runs `os.environ.setdefault("CHROMA_BACKEND", "triton")`
before anything imports `chroma`. A user who sets `CHROMA_BACKEND=cuda` (with the
`cuda` extra installed) gets CUDA Chroma. If `chroma.sim` was imported before
`chroma_lar` with a different backend, `chroma_lar` warns once that the default
could not apply. `pyrat` imports `chroma_lar` before `chroma`, so every macro gets
the default.

The generated job scripts (`jobs/submit_waveform_jobs.py`) export
`TRITON_CACHE_DIR` next to the existing `PYCUDA_CACHE_DIR`, so compiled kernels go
to the job's scratch directory and not to `$HOME`.

## Unweighted LUT generation

LUT generation must use unweighted photons: `yield_counts` are integer photon
counts and `quantiles` are quantiles of individual hit times. `pyrat` passes
`use_weights=False` to `sim.simulate` explicitly (both backends accept it), so a
changed default cannot switch LUT runs to weighted mode. The README states this,
and that `CHROMA_TRITON=roulette=<w>` only acts in weighted mode.

## Containers

`installation/chroma3.lar-plib/Dockerfile` and `Singularity` both:

- start from `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (Singularity moves
  from cu11.8);
- install uv and run `uv pip install --system` on chroma-lar, which brings
  chroma-lite, TriChroma and photonlib; no PyCUDA;
- drop the hand clone of chroma-lite and its `sed` edits (the `camera.py` fix is
  kept only if chroma-lite `master` still needs it);
- keep the analysis extras (siren-lartpc, sirentv, gdown) and, in Singularity,
  the GL libraries.

The Singularity image is built on S3DF with `apptainer build --fakeroot`, with
the build, cache and image under `/sdf/data/neutrino/youngsam/`.

## Locations

All builds live on `/sdf/data/neutrino/youngsam/`, not `/sdf/group/`: the
worktree (`chroma-lar-trichroma`), the uv environment, `UV_CACHE_DIR`,
`APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR`, the image and the Triton cache.

## Testing

1. CPU pytest (`tests/`): importing `chroma_lar` sets `CHROMA_BACKEND=triton`
   when unset and keeps an existing value; `chroma.sim.Simulation` is then
   TriChroma's class.
2. Install time: a cold `uv sync` (empty `UV_CACHE_DIR`), timed.
3. GPU, on a turing (RTX 2080 Ti) and an ampere (A100) node. TriChroma's
   production mode fixes bugs of CUDA Chroma, so it is expected to differ from
   it slightly; its `legacy` option and its exact (tape) mode turn those fixes
   off. The CUDA reference is `chroma-lar[cuda]` installed inside the new image
   with `CHROMA_BACKEND=cuda`, i.e. the same chroma-lite revision, which has the
   tape recording hook.
   - Bitwise: record a small CUDA run of the reflect3wires detector
     (`CHROMA_TRITON_TAPE=record:<dir>`, a few voxels' photon bombs), replay it
     with TriChroma (`CHROMA_TRITON_TAPE=replay:<dir>`), and require
     `trichroma.utils.compare_events` to report identical events.
   - Statistical: run `macros/waveform_map_pyrat_quantile.py` on about 8 voxels
     (near and far from the PMTs, a few million photons each), with the audit
     mode storing raw hits for every voxel, with CUDA Chroma and with TriChroma
     `CHROMA_TRITON=legacy`. Compare per voxel and channel: yields
     (two-proportion z) and hit-time distributions from the raw hits (KS
     distance against the DKW bound), with `trichroma.utils` helpers. Pass: no
     |z| > 5 across all tests, the z values have mean near 0 and standard
     deviation near 1, and no KS failure at alpha = 1e-6.
   - Production: the same comparison with TriChroma's default (production)
     mode, reported (largest |z|, z mean and spread, per-voxel yield ratio)
     without a pass/fail threshold, since the bug fixes may shift results.
   - The production TriChroma run inside the built Singularity image succeeds.
   - Report photons/s for both backends on both GPUs.

## Out of scope

Pushing the branch or opening a PR (waits for the user), the NERSC Shifter
image, changes to chroma-lite or TriChroma, and the viewer (`chroma.camera`
still needs the `cuda` extra and pygame).
