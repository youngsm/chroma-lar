# TriChroma as chroma-lar's default backend

Date: 2026-10-14. Branch: `trichroma-default`, from `origin/trichroma-fast-waveform`.
Depends on: chroma-lite's `merge-trichroma` spec
(`docs/superpowers/specs/2026-10-14-merge-trichroma-design.md` in chroma-lite),
which merges TriChroma into chroma-lite as `chroma.triton`, makes Triton the
default backend, and renames the repository and distribution `trichroma`.

## Goal

chroma-lar runs its simulations on TriChroma (Triton) by default. Installing it
takes a few minutes with uv and compiles nothing; PyCUDA is an optional extra.
The CUDA backend stays available, LUT generation stays unweighted, and every
LUT records how it was made.

## What exists

- After the merge, the `trichroma` distribution (import name `chroma`) provides
  Chroma's API and both backends. `chroma.sim.Simulation` is the Triton
  engine unless `CHROMA_BACKEND=cuda`; PyCUDA is its `cuda` extra.
- chroma-lar `origin/trichroma-fast-waveform` makes the waveform macros draw
  photons on the GPU with `trichroma.sources` when it is importable.
- chroma-lar has a `setup.py` with no dependencies, a loose `requirements.txt`,
  and container recipes that clone chroma-lite by hand.

## Packaging

`pyproject.toml` (setuptools backend) replaces `setup.py` and
`requirements.txt`. It finds the same packages `find_packages()` found
(`chroma_lar`, `data`, `utils`). `requires-python = ">=3.10"`.

Dependencies, as PEP 508 direct references so both pip and uv resolve them:

| Requirement | Why |
|---|---|
| `trichroma @ git+https://github.com/youngsm/trichroma.git` | Chroma's API and the Triton backend |
| `photonlib @ git+https://github.com/CIDeR-ML/photonlib.git` | `VoxelMeta` in the macros |
| `torch==2.5.1` | TriChroma's validated stack; brings `triton==3.1.0`; PyPI wheels are cu124 |
| numpy, scipy, h5py, trimesh, PyYAML, pydantic, rich, tqdm, matplotlib, fire | used by the package, macros and job scripts |

Extra `cuda = ["trichroma[cuda] @ git+https://github.com/youngsm/trichroma.git"]`
adds PyCUDA; nothing else compiles.

Until the GitHub rename, the `trichroma` URL above still names the old
standalone repository, so development points the dependency at the merged
chroma-lite branch (`git+https://github.com/youngsm/chroma-lite.git@merge-trichroma`,
or the local checkout). The committed `pyproject.toml` uses the final URL once
the rename is done.

`uv.lock` is committed. Install: `uv sync` (add `--extra cuda` for PyCUDA).
`chroma_lar_env.sh` and `pyrat` keep working as before.

## Default backend

chroma itself defaults to Triton, so chroma-lar sets no environment variable.
`CHROMA_BACKEND=cuda` (with the `cuda` extra) runs CUDA Chroma.

The waveform macros import GPU photon sources as `from chroma.triton import
sources`; with `CHROMA_BACKEND=cuda` they keep drawing numpy photons.

The generated job scripts (`jobs/submit_waveform_jobs.py`) export
`TRITON_CACHE_DIR` next to the existing `PYCUDA_CACHE_DIR`, so compiled kernels go
to the job's scratch directory and not to `$HOME`.

## Production physics

LUT macros run TriChroma's production mode, which fixes CUDA Chroma bugs. On
chroma-lar's TPC the wire-plane fix alone raises detected light by about 18%,
so new LUTs are not comparable to b04/b05. `CHROMA_TRITON=legacy` restores CUDA
Chroma's behaviour for like-for-like comparisons.

## Unweighted LUT generation

LUT generation must use unweighted photons: `yield_counts` are integer photon
counts and `quantiles` are quantiles of individual hit times. `pyrat` passes
`use_weights=False` to `sim.simulate` explicitly (both backends accept it), so a
changed default cannot switch LUT runs to weighted mode. The README states this,
and that `CHROMA_TRITON=roulette=<w>` only acts in weighted mode.

## LUT provenance

A helper `chroma_lar.provenance.run_attrs()` returns a flat dict of strings and
numbers describing the run:

- `backend` (`triton` or `cuda`), `chroma_triton` (the `CHROMA_TRITON` value,
  empty for production), `chroma_triton_tape`;
- `trichroma_version` and `trichroma_commit`, and `chroma_lar_version` and
  `chroma_lar_commit`, taken from the installed distributions' metadata (the
  commit of a git install) or from `git` for an editable checkout, empty when
  unknown, plus `chroma_lar_dirty` for a checkout with local changes;
- `use_weights` (always 0), `torch_version`, `triton_version`, `gpu_name`.

The waveform-map macros (`waveform_map_pyrat.py`,
`waveform_map_pyrat_32bit.py`, `waveform_map_pyrat_quantile.py`) write these
into their output file's root attributes, next to the attributes they already
write. Consolidation scripts that copy root attributes keep them.

## Containers

`installation/chroma3.lar-plib/Dockerfile` and `Singularity` both:

- start from `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (Singularity moves
  from cu11.8);
- install uv and run `uv pip install --system` on chroma-lar, which brings
  TriChroma and photonlib; no PyCUDA;
- drop the hand clone of chroma-lite and its `sed` edits (the `camera.py` fix is
  kept only if it is still needed);
- keep the analysis extras (siren-lartpc, sirentv, gdown) and, in Singularity,
  the GL libraries.

The Singularity image is built on S3DF with `apptainer build --fakeroot`, with
the build, cache and image under `/sdf/data/neutrino/youngsam/`.

## Locations

All builds live on `/sdf/data/neutrino/youngsam/`, not `/sdf/group/`: the
worktree (`chroma-lar-trichroma`), the uv environment, `UV_CACHE_DIR`,
`APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR`, the image and the Triton cache.

## Testing

1. CPU pytest (`tests/`): `import chroma_lar` works on a node without a GPU;
   `chroma.sim.Simulation` is the Triton engine's class with
   `CHROMA_BACKEND` unset; `run_attrs()` returns every key above, with
   `use_weights` 0.
2. Install time: a cold `uv sync` (empty `UV_CACHE_DIR`), timed.
3. GPU, on a turing (RTX 2080 Ti) and an ampere (A100) node. TriChroma's
   production mode fixes bugs of CUDA Chroma, so it is expected to differ from
   it; its `legacy` option and its exact (tape) mode turn those fixes off. The
   CUDA reference is `chroma-lar[cuda]` installed inside the new image with
   `CHROMA_BACKEND=cuda`, i.e. the same TriChroma revision, which has the tape
   recording hook.
   - Bitwise: record a small CUDA run of the reflect3wires detector
     (`CHROMA_TRITON_TAPE=record:<dir>`, a few voxels' photon bombs), replay it
     with the Triton backend (`CHROMA_TRITON_TAPE=replay:<dir>`), and require
     `chroma.triton.utils.compare_events` to report identical events.
   - Statistical: run `macros/waveform_map_pyrat_quantile.py` on about 8 voxels
     (near and far from the PMTs, a few million photons each), with the audit
     mode storing raw hits for every voxel, with CUDA Chroma and with
     `CHROMA_TRITON=legacy`. Compare per voxel and channel: yields
     (two-proportion z) and hit-time distributions from the raw hits (KS
     distance against the DKW bound), with `chroma.triton.utils` helpers. Pass:
     no |z| > 5 across all tests, the z values have mean near 0 and standard
     deviation near 1, and no KS failure at alpha = 1e-6.
   - Production: the same comparison with the default (production) mode,
     reported (largest |z|, z mean and spread, per-voxel yield ratio) without a
     pass/fail threshold; the yield ratio is expected near the guide's +18%.
   - The output files carry the provenance attributes, with the right backend
     and options for each run.
   - The production run inside the built Singularity image succeeds.
   - Report photons/s for both backends on both GPUs.

## Out of scope

Pushing the branch or opening a PR (waits for the user), the NERSC Shifter
image, and the viewer (`chroma.camera` still needs the `cuda` and `viz` extras).
The merge itself is the chroma-lite spec's work.
