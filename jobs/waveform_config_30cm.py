# Detector and voxel configuration
config = {
    # Detector dimensions (mm)
    "detector_x_range": (-2310, 0),  # Only simulating half (mirrored)
    "detector_y_range": (-2160, 2160),
    "detector_z_range": (-2160, 2160),
    # Voxel parameters
    "voxel_size": 300,  # mm
    # Simulation parameters
    "nphotons": 200_000,
    "detector_config": "detector_config_reflect_reflect3wires",
    # Job parameters
    "time_per_voxel": 15,  # seconds per voxel
    "max_job_time": 3 * 60 * 60,  # 3 hours in seconds
    "output_dir": "waveform_maps",
    # Singularity container
    "container": "/sdf/home/y/youngsam/sw/dune/sim/chroma-lar/installation/chroma3.lar-plib/chroma.simg",
    "partition": "ampere",
    "account": "neutrino:cider-nu",
}
