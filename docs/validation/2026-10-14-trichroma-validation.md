# TriChroma validation for chroma-lar

Date: 2026-10-14.
Code: chroma-lar `0232ee1` (branch `feat/trichroma-default`) with TriChroma `d2b2870` (chroma-lite branch `feat/merge-trichroma`), in the Singularity image built from `installation/chroma3.lar-plib/Singularity` (PyTorch 2.5.1, CUDA 12.4, Triton 3.1.0).
The CUDA reference is the same image plus PyCUDA 2026.1 (`CHROMA_BACKEND=cuda`).
Detector: `detector_config_reflect_reflect3wires`, 450 nm.
Driver: `scripts/validation/compare_backends.sh`; outputs in `/sdf/data/neutrino/youngsam/validation/trichroma-default/`.

## Results

| Check | A100 (driver 575.57) | RTX 2080 Ti (driver 575.57) |
|---|---|---|
| Bitwise: CUDA Chroma recorded, replayed by `chroma.triton` (`tape.verify all --fixture reflect3wires`) | all outputs equal | all outputs equal |
| Legacy (`CHROMA_TRITON=legacy`) against CUDA, 8 voxels × 2M photons, 648 voxel-channel tests | passed: max \|z\| 3.56, z mean 0.045, z spread 1.041, 0 KS failures | passed: max \|z\| 3.08, z mean -0.033, z spread 1.026, 0 KS failures |
| Legacy yield ratio (TriChroma / CUDA) | 1.003 | 0.998 |
| Production (default) against CUDA, informational | yield ratio 1.185, z mean 3.59, max \|z\| 12.9, 1 KS failure | yield ratio 1.183, z mean 3.58, max \|z\| 12.1, 1 KS failure |
| Provenance attributes | correct backend, `CHROMA_TRITON`, commits, `use_weights` 0, GPU name in all three files | same |

Production mode detects about 18% more light than CUDA Chroma, as expected from TriChroma's wire-plane fix; legacy mode agrees with CUDA Chroma within statistics.

## Throughput

`benchmarks/simulate_throughput.py` (one point source, photons made on the host, 450 nm), best of three calls:

| Backend | A100 | RTX 2080 Ti |
|---|---|---|
| CUDA Chroma, 10M photons | 3.7M photons/s | 1.5M photons/s |
| TriChroma, 30M photons | 97M photons/s | 38M photons/s |

LUT generation on an A100 (24 voxels × 30M photons, photons drawn on the GPU, same job):

| Workload | Photons/s |
|---|---|
| Engine only, no hits returned | 143M (1 voxel per launch), 145M (3 per launch) |
| Hits returned and reduced to quantiles on the host | 134M to 138M |
| Full `waveform_map_pyrat_quantile.py` macro | 124M steady (118M including the first voxel) |

## Installation

- Cold `uv sync --frozen` (empty cache, node-local disk): 34 s, 69 packages, nothing compiled.
- Singularity image build on a CPU node (`apptainer build --fakeroot`): 7 min 10 s.
