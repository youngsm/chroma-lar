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
        assert "macros/waveform_map_pyrat_quantile.py" in run  # the default macro
