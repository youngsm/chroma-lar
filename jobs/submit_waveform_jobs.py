#!/usr/bin/env python3
import os
import sys
import math
import argparse
import subprocess
import datetime
import importlib.util
import inspect

def calculate_batch_size(config):
    """Calculate how many voxels can be processed in the given job time"""
    return int(config["max_job_time"] / config["time_per_voxel"])

def calculate_total_voxels(config):
    """Calculate the total number of voxels in the detector"""
    x_size = abs(config["detector_x_range"][1] - config["detector_x_range"][0])
    y_size = abs(config["detector_y_range"][1] - config["detector_y_range"][0])
    z_size = abs(config["detector_z_range"][1] - config["detector_z_range"][0])
    
    nx = (x_size // config["voxel_size"])
    ny = (y_size // config["voxel_size"])
    nz = (z_size // config["voxel_size"])

    return nx * ny * nz, nx, ny, nz

def calculate_job_time(batch_size, time_per_voxel):
    """Calculate expected job time for a given batch size"""
    return batch_size * time_per_voxel / 3600

def calculate_voxel_indices(job_id, batch_size, config):
    """Calculate the range of voxel indices this job should process"""
    total_voxels, _, _, _ = calculate_total_voxels(config)
    
    # Calculate start and end indices for this job
    start_idx = job_id * batch_size
    end_idx = min((job_id + 1) * batch_size, total_voxels)

    return start_idx, end_idx

def main():
    parser = argparse.ArgumentParser(description='Submit waveform mapping jobs')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--batch-size', type=int, help='Override calculated batch size')
    parser.add_argument('--throttle', type=int, help='Throttle the number of concurrent jobs', default=-1)
    parser.add_argument('--run-job', type=int, help='Run a specific job ID locally for testing', default=None)
    parser.add_argument('--config', type=str, help='Path to the configuration file', default='./waveform_config_3cm.py')
    parser.add_argument('--site', type=str, help='Slurm parameter set (slac or perlmutter) override in the config file', default=None)
    args = parser.parse_args()
    
    # load the configuration file
    spec = importlib.util.spec_from_file_location("waveform_config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = inspect.getattr_static(config_module, "config")
    site_config = inspect.getattr_static(config_module, "site")

    site = args.site if args.site else config.get("site", "slac")
    site_config = site_config.get(site)
    slurm_config = site_config.get("slurm")

    # calculate batch size (# positions / job) and total jobs
    batch_size = args.batch_size if args.batch_size else calculate_batch_size(config)
    total_voxels, nx, ny, nz = calculate_total_voxels(config)
    total_jobs = math.ceil(total_voxels / batch_size)
    
    # try running a specific job locally for testing
    if args.run_job is not None:
        job_id = args.run_job
        start_idx, end_idx = calculate_voxel_indices(job_id, batch_size, config)
        output_filename = f"{site_config['output_dir']}/waveform_map_job_{job_id}.h5"
        
        # Get the absolute path to waveform_map_pyrat.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        waveform_map_script = os.path.join(current_dir, "..", "macros", "waveform_map_pyrat.py")
        pyrat_script = os.path.join(current_dir, "..", "pyrat")
        
        cmd = f"""
        PYCUDA_CACHE_DIR=/lscratch singularity exec --nv -B /lscratch,/sdf {config["container"]} \\
        /opt/conda/bin/python {pyrat_script} {waveform_map_script} \\
            -s detector_config {config["detector_config"]} \\
            -es voxel_ranges "({config["detector_x_range"]}, {config["detector_y_range"]}, {config["detector_z_range"]})" \\
            -es voxel_size {config["voxel_size"]} \\
            -es nphotons {config["nphotons"]} \\
            -es voxel_index_start {start_idx} \\
            -es batch_size {end_idx - start_idx} \\
            -s output_filename {output_filename}
        """
        
        print(f"Running job {job_id} locally:")
        print(f"  Voxel indices: {start_idx} to {end_idx-1}")
        print(f"  Output file: {output_filename}")
        print(f"  Command: {cmd}")
        
        os.system(cmd)
        return
    
    # print statistics summary
    print("\n=== WAVEFORM MAPPING JOB SUBMISSION ===\n")
    print("Configuration Summary:")
    print(f"  Detector dimensions: X={config['detector_x_range']}, Y={config['detector_y_range']}, Z={config['detector_z_range']}")
    print(f"  Voxel size: {config['voxel_size']}mm")
    print(f"  Voxel grid: {nx} × {ny} × {nz} = {total_voxels} voxels")
    print(f"  Photons per voxel: {config['nphotons']:,}")
    print(f"  Detector config: {config['detector_config']}")
    print(f"  Output directory: {site_config['output_dir']}")
    print(f"  Container CMD: {site_config['container_cmd']}")
    
    print("\nJob Distribution:")
    print(f"  Estimated time per voxel: {config['time_per_voxel']} seconds")
    print(f"  Maximum job time: {config['max_job_time']/3600:.1f} hours")
    print(f"  Batch size: {batch_size} voxels per job")
    print(f"  Estimated job runtime: {calculate_job_time(batch_size, config['time_per_voxel']):.1f} hours")
    print(f"  Total jobs required: {total_jobs}")
    print(f"  Estimated total computation time: {total_jobs * batch_size * config['time_per_voxel']/3600:.1f} GPU-hours")
    
    # create output directory if it doesn't exist
    for d in [site_config['output_dir'],os.path.dirname(slurm_config['output']),os.path.dirname(slurm_config['error'])]:
        if not os.path.exists(d):
            os.makedirs(d)

    if args.throttle > 0:
        throttle_str = f"%{args.throttle}"
    else:
        throttle_str = ""
    
    # get the absolute path to the waveform_map_pyrat.py script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    waveform_map_script = os.path.join(current_dir, "..", "macros", "waveform_map_pyrat.py")
    pyrat_script = os.path.join(current_dir, "..", "pyrat")
    
    slurm_limit = str(datetime.timedelta(seconds=config["max_job_time"]+config["slurm_max_job_time_buffer"])) # format: HH:MM:SS
    
    # create a single slurm job array submission script
    array_script = f"""#!/bin/bash
#SBATCH --job-name=wfmap_array
#SBATCH --time={slurm_limit}
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --array=0-{total_jobs - 1}{throttle_str}
"""

    for skey,sval in slurm_config.items():
        # ensure no duplication
        flag = f'#SBATCH --{skey}'
        if flag in array_script:
            raise ValueError(f"Duplicate SLURM setting found: {flag}")
        array_script += f"{flag}={sval}\n"

    array_script += f"""

# Calculate voxel indices based on SLURM_ARRAY_TASK_ID
job_id=$SLURM_ARRAY_TASK_ID
start_idx=$((job_id * {batch_size}))
end_idx=$(( (job_id + 1) * {batch_size} ))

# Make sure end_idx doesn't exceed total voxels
total_voxels={total_voxels}
if [ $end_idx -gt $total_voxels ]; then
    end_idx=$total_voxels
fi

batch_size=$((end_idx - start_idx))
output_file="{site_config['output_dir']}/waveform_map_job_${{job_id}}.h5"

echo "Processing job $job_id: voxels $start_idx to $((end_idx - 1))"
echo "Output file: $output_file"

# Run the pyrat command directly
export PYCUDA_CACHE_DIR=/lscratch
{site_config['container_cmd']} \\
    /opt/conda/bin/python {pyrat_script} {waveform_map_script} \\
    --set detector_config {config["detector_config"]} \\
    --evalset voxel_ranges "({config['detector_x_range']}, {config['detector_y_range']}, {config['detector_z_range']})" \\
    --evalset voxel_size {config["voxel_size"]} \\
    --evalset nphotons {config["nphotons"]} \\
    --evalset voxel_index_start $start_idx \\
    --evalset batch_size $batch_size \\
    --set output_filename $output_file
"""
    
    # Save the array job script
    array_script_path = f"{site_config['output_dir']}/wfmap_array_job.sh"
    with open(array_script_path, "w") as f:
        f.write(array_script)
    
    print("\nJob Array Submission:")
    if args.dry_run:
        print("  DRY RUN")
        print(f"  Would create job array script: {array_script_path}")
        print(f"  Would submit: sbatch {array_script_path}")
        print(f"  Job array would create {total_jobs} tasks (0-{total_jobs-1})")
    else:
        print(f"  Created job array script: {array_script_path}")
        print(f"  Submitting job array with {total_jobs} tasks...")
        result = subprocess.run(f"sbatch {array_script_path}", shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"  Success! Job array submitted with ID: {job_id}")
        else:
            print(f"  Error submitting job array: {result.stderr}")
    
    print(f"\nJob array will process all {total_voxels} voxels using {total_jobs} tasks")

if __name__ == "__main__":
    main()
