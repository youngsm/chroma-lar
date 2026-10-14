"""Statistical comparison of two waveform-map runs (audit hits for every voxel)."""

import h5py
import numpy as np
import pytest

from chroma_lar import validation

NPHOTONS = 1_000_000
NCHAN = 20
VOXELS = list(range(100, 108))


def _write(path, seed, yield_scale=1.0, time_shift=0.0, voxels=VOXELS):
    rng = np.random.default_rng(seed)
    lam = np.linspace(200, 2000, NCHAN) * yield_scale
    t_all, ch_all, offsets, nhits = [], [], [], []
    offset = 0
    for _ in voxels:
        counts = rng.poisson(lam)
        ch = np.repeat(np.arange(NCHAN), counts).astype(np.uint16)
        t = (rng.exponential(10.0, ch.size) + ch * 0.5 + time_shift).astype(np.float32)
        t_all.append(t); ch_all.append(ch); offsets.append(offset); nhits.append(ch.size)
        offset += ch.size
    with h5py.File(path, "w") as f:
        f.attrs["nphotons_generated"] = NPHOTONS
        f.attrs["num_output_channels"] = NCHAN
        f["audit_t"] = np.concatenate(t_all)
        f["audit_channel"] = np.concatenate(ch_all)
        f["audit_voxel_id"] = np.asarray(voxels, dtype=np.uint32)
        f["audit_offset"] = np.asarray(offsets, dtype=np.uint64)
        f["audit_nhits"] = np.asarray(nhits, dtype=np.uint32)
    return path


def test_load_hits(tmp_path):
    hits, nphotons, nchan = validation.load_hits(_write(tmp_path / "a.h5", 1))
    assert (nphotons, nchan) == (NPHOTONS, NCHAN)
    assert sorted(hits) == VOXELS
    t, ch = hits[100]
    assert t.shape == ch.shape


def test_same_distribution_passes(tmp_path):
    result = validation.compare(_write(tmp_path / "a.h5", 1), _write(tmp_path / "b.h5", 2))
    assert result["passed"], result
    assert result["tests"] == len(VOXELS) * NCHAN
    assert abs(result["z_mean"]) < 0.4 and 0.7 < result["z_std"] < 1.3
    # each voxel has ~22k hits per run: the ratio's standard error is ~1%
    assert all(abs(r - 1) < 0.05 for r in result["yield_ratio"].values())


def test_more_light_fails_and_reports_ratio(tmp_path):
    result = validation.compare(_write(tmp_path / "a.h5", 1), _write(tmp_path / "b.h5", 2, yield_scale=1.18))
    assert not result["passed"]
    assert all(abs(r - 1.18) < 0.06 for r in result["yield_ratio"].values())
    assert result["z_mean"] > 1  # every channel is brighter


def test_later_times_fail_ks(tmp_path):
    result = validation.compare(_write(tmp_path / "a.h5", 1), _write(tmp_path / "b.h5", 2, time_shift=3.0))
    assert not result["passed"]
    assert result["ks_failures"]


def test_missing_voxel_fails(tmp_path):
    result = validation.compare(_write(tmp_path / "a.h5", 1), _write(tmp_path / "b.h5", 2, voxels=VOXELS[:-1]))
    assert not result["passed"]
    assert result["missing"] == [VOXELS[-1]]


def test_channel_count_mismatch_raises(tmp_path):
    a = _write(tmp_path / "a.h5", 1)
    b = _write(tmp_path / "b.h5", 2)
    with h5py.File(b, "a") as f:
        f.attrs["num_output_channels"] = NCHAN + 1
    with pytest.raises(ValueError):
        validation.compare(a, b)


def test_cli_exit_code(tmp_path):
    a = _write(tmp_path / "a.h5", 1)
    assert validation.main([str(a), str(_write(tmp_path / "b.h5", 2)), "--json", str(tmp_path / "r.json")]) == 0
    assert validation.main([str(a), str(_write(tmp_path / "c.h5", 3, yield_scale=1.18))]) == 1
    assert (tmp_path / "r.json").exists()
