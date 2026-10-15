"""Statistical comparison of two waveform-map runs of the same voxels.

Both files come from ``macros/waveform_map_pyrat_quantile.py`` with raw hits
kept for every voxel (``audit_every_n_voxels=1``). For every voxel and channel
with at least ``min_hits`` hits in the two runs together, the yields are
compared with a two-proportion z test (positive when B is brighter, like the
yield ratio B/A) and the hit times with a two-sample KS
test against its DKW bound at ``alpha``. The runs agree (``passed``) when no
|z| exceeds ``max_sigma``, the z values have mean near 0 and spread near 1
(within five standard errors), no KS test fails, and both runs have the same
voxels.

Usage: python -m chroma_lar.validation A.h5 B.h5 [--json OUT] [--max-sigma 5]
"""

import argparse
import json
import sys

import h5py
import numpy as np

from chroma.triton.utils import ks_test, two_proportion_z


def load_hits(path):
    """``({voxel_id: (times, channels)}, nphotons per voxel, channels)`` of a run's raw hits."""
    with h5py.File(path, "r") as f:
        nphotons = int(f.attrs["nphotons_generated"])
        nchan = int(f.attrs["num_output_channels"])
        voxel_ids = f["audit_voxel_id"][:]
        offsets = f["audit_offset"][:]
        nhits = f["audit_nhits"][:]
        times = f["audit_t"][:]
        channels = f["audit_channel"][:]
    hits = {int(v): (times[o:o + n], channels[o:o + n]) for v, o, n in zip(voxel_ids, offsets, nhits)}
    return hits, nphotons, nchan


def compare(path_a, path_b, max_sigma=5.0, alpha=1e-6, min_hits=25):
    """Compare run B against run A; see the module docstring."""
    hits_a, n_a, nchan = load_hits(path_a)
    hits_b, n_b, nchan_b = load_hits(path_b)
    if nchan != nchan_b:
        raise ValueError("the runs have %d and %d output channels" % (nchan, nchan_b))
    voxels = sorted(set(hits_a) & set(hits_b))
    z_values, ks_failures, ratios, worst = [], [], {}, None
    for v in voxels:
        times_a, chan_a = hits_a[v]
        times_b, chan_b = hits_b[v]
        k_a = np.bincount(chan_a, minlength=nchan)
        k_b = np.bincount(chan_b, minlength=nchan)
        ratios[v] = float((k_b.sum() / n_b) / (k_a.sum() / n_a)) if k_a.sum() else float("nan")
        for c in np.nonzero(k_a + k_b >= min_hits)[0]:
            z = two_proportion_z(int(k_b[c]), n_b, int(k_a[c]), n_a)  # > 0: B is brighter
            z_values.append(z)
            if worst is None or abs(z) > abs(worst["z"]):
                worst = {"z": float(z), "voxel": v, "channel": int(c)}
            ks = ks_test(times_a[chan_a == c], times_b[chan_b == c], alpha=alpha)
            if ks is not None and not ks["pass"]:
                ks_failures.append({"voxel": v, "channel": int(c), "distance": ks["distance"],
                                    "bound": ks["bound"]})
    z = np.asarray(z_values, dtype=np.float64)
    result = {
        "voxels": len(voxels),
        "missing": sorted(set(hits_a) ^ set(hits_b)),
        "tests": int(z.size),
        "z_max": float(np.abs(z).max()) if z.size else 0.0,
        "z_mean": float(z.mean()) if z.size else 0.0,
        "z_std": float(z.std()) if z.size else 0.0,
        "worst": worst,
        "ks_failures": ks_failures,
        "yield_ratio": {str(v): r for v, r in ratios.items()},
    }
    n = max(z.size, 1)
    result["passed"] = bool(
        z.size
        and result["z_max"] <= max_sigma
        and abs(result["z_mean"]) <= 5 / np.sqrt(n)
        and abs(result["z_std"] - 1) <= 5 / np.sqrt(2 * n)
        and not ks_failures
        and not result["missing"]
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--json", help="write the full result here")
    parser.add_argument("--max-sigma", type=float, default=5.0)
    args = parser.parse_args(argv)
    result = compare(args.a, args.b, max_sigma=args.max_sigma)
    summary = {k: result[k] for k in ("passed", "voxels", "missing", "tests", "z_max", "z_mean", "z_std", "worst")}
    summary["ks_failures"] = len(result["ks_failures"])
    ratios = list(result["yield_ratio"].values())
    summary["yield_ratio_mean"] = float(np.nanmean(ratios)) if ratios else float("nan")
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
