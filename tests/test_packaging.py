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
