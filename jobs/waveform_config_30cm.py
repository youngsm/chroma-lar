import os

"""
TriChroma on an A100 runs ~124M photons/s on the 3 cm grid; 15M photons per voxel
is estimated at ~0.12 s per voxel (not measured on this grid).
"""
# Detector and voxel configuration
config = {
    # Detector dimensions (mm)
    "detector_x_range": (-2310, 0),  # Only simulating half (mirrored)
    "detector_y_range": (-2160, 2160),
    "detector_z_range": (-2160, 2160),
    # Voxel parameters
    "voxel_size": 300,  # mm
    # Simulation parameters
    "nphotons": 15_000_000,
    "detector_config": "detector_config_reflect_reflect3wires",
    "macro": "waveform_map_pyrat_quantile.py",
    # Job parameters
    "time_per_voxel": 0.15,  # seconds per voxel (estimate)
    "max_job_time": 3 * 60 * 60,  # 3 hours in seconds
    "slurm_max_job_time_buffer": 2 * 60,  # 2 minutes in seconds
    "site": "slac",
}


site = dict(
    slac=dict(
        work_dir="$LSCRATCH",
        # no container_cmd: jobs run this checkout's uv environment (`uv sync` first)
        output_dir=f"/sdf/data/neutrino/{os.environ['USER']}/prod_chroma_lar/waveform_map",
        slurm=dict(
            partition='ampere',
            account='neutrino:cider-nu',
            output=f"/sdf/data/neutrino/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            error=f"/sdf/data/neutrino/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
        ),
    ),
    perlmutter=dict(
        work_dir="$PSCRATCH",
        # predates TriChroma (runs CUDA Chroma): rebuild from installation/chroma3.lar-plib/Dockerfile
        container_cmd="shifter --image=deeplearnphysics/simlar:larchroma-2025-07-18",
        output_dir=f"/global/cfs/cdirs/dune/users/{os.environ['USER']}/prod_chroma_lar/waveform_map",
        slurm=dict(
            account='dune',
            output=f"/global/cfs/cdirs/dune/users/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            error=f"/global/cfs/cdirs/dune/users/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            qos='shared',
            constraint='gpu',
        )
    )
)
