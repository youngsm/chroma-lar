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
mkdir -p "$PYCUDA_CACHE_DIR"
# $HOME is bound into the container: keep ~/.local installs from shadowing the image's chroma.
export PYTHONNOUSERSITE=1
# A login node's TMPDIR may not exist on this node; nvcc needs one that does.
export TMPDIR=/lscratch/$USER/tmp-${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
run() { apptainer exec --nv -B /sdf,/lscratch "$SIF" bash -c "cd $OUT && $*"; }
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
