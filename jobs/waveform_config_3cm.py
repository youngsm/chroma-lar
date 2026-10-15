import os

"""
TriChroma on an A100, 30M photons per voxel: 0.24 s per voxel (124M photons/s)
--> 1596672 voxels = ~107 GPU-hours.
"""

# Detector and voxel configuration
config = {
    # Detector dimensions (mm)
    "detector_x_range": (-2310, 0),  # Only simulating half (mirrored)
    "detector_y_range": (-2160, 2160),
    "detector_z_range": (-2160, 2160),
    # Voxel parameters
    "voxel_size": 30,  # mm
    # Simulation parameters
    "nphotons": 30_000_000,
    "detector_config": "detector_config_reflect_reflect3wires",
    "macro": "waveform_map_pyrat_quantile.py",
    # Job parameters
    "time_per_voxel": 0.3,  # seconds per voxel
    "max_job_time": 8 * 60 * 60,  # 8 hours in seconds
    "slurm_max_job_time_buffer": 2 * 60 * 60,  # 2 hours in seconds
    # Site-specific configuration
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
        output_dir=f"/global/cfs/cdirs/m5238/users/{os.environ['USER']}/prod_chroma_lar/waveform_map",
        slurm=dict(
            account='m5238_g',
            output=f"/global/cfs/cdirs/m5238/users/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            error=f"/global/cfs/cdirs/m5238/users/{os.environ['USER']}/prod_chroma_lar/logs/wfmap_%A_%a.log",
            qos='shared',
            constraint='gpu',
        )
    )
)
