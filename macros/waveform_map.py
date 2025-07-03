import os
import logging
from chroma.log import logger
logger.setLevel(logging.DEBUG)
import numpy as np
import h5py

from chroma_lar.geometry import build_detector_from_config
from chroma.sim import Simulation
from chroma.event import Photons

def create_photon_bomb(nphotons, pos, voxel_size=30, wavelength=128) -> Photons:
    costheta = np.random.random(nphotons)*2-1
    sintheta = np.sqrt(1-np.square(costheta))
    phi = np.random.random(nphotons)*2*np.pi
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    pdir = np.transpose([sintheta*cosphi, sintheta*sinphi, costheta])

    costheta = np.random.random(nphotons)*2-1
    sintheta = np.sqrt(1-np.square(costheta))
    phi = np.random.random(nphotons)*2*np.pi
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    rand_unit = np.transpose([sintheta*cosphi, sintheta*sinphi, costheta])
    ppol = np.cross(pdir, rand_unit)
    ppol = ppol / np.linalg.norm(ppol, ord=2, axis=1, keepdims=True)

    if type(wavelength) is tuple:
        pwavelength = (
            np.random.random(nphotons) * (wavelength[1] - wavelength[0]) + wavelength[0]
        )
    else:
        pwavelength = np.tile(wavelength, nphotons)

    x = np.random.random(nphotons)*voxel_size-voxel_size/2+pos[0]
    y = np.random.random(nphotons)*voxel_size-voxel_size/2+pos[1]
    z = np.random.random(nphotons)*voxel_size-voxel_size/2+pos[2]
    ppos = np.transpose([x,y,z])

    return Photons(pos=ppos, dir=pdir, pol=ppol, wavelengths=pwavelength)

class H5Writer:
    def __init__(self, filename, num_pmts=162, num_ticks=1000, max_time=100):
        self.filename = filename
        self.num_pmts = num_pmts
        self.num_ticks = num_ticks
        self.max_time = max_time

        if os.path.exists(filename):
            raise FileExistsError(f"File {filename} already exists!")
        self.file = h5py.File(filename, "w")
        self.file.create_dataset(
            "counts",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint16,
            chunks=True
        )
        self.file.create_dataset(
            "voxel_ids",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint32,
            chunks=True
        )
        
    def save(self, ev, voxel_id):
        times = ev.flat_hits.t
        channels = ev.flat_hits.channel
        counts = np.histogram2d(
            channels,
            times,
            bins=(self.num_pmts, self.num_ticks),
            range=((0, self.num_pmts), (0, self.max_time)),
        )[0].flatten()
        counts = counts.astype(np.uint16)
        self.file["counts"].resize(self.file["counts"].shape[0] + 1, axis=0)
        self.file["counts"][-1] = counts
        self.file["voxel_ids"].resize(self.file["voxel_ids"].shape[0] + 1, axis=0)
        self.file["voxel_ids"][-1] = voxel_id
        self.file.flush()

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()

def run_waveform_map(
    output_filename,
    detector_config="detector_config_reflect_reflect3wires",
    voxel_shape=(77, 144, 144),
    voxel_ranges=((-2310, 0), (-2160, 2160), (-2160, 2160)),
    voxel_index_start=0,
    batch_size=720,
    nphotons=200_000,
    voxel_size=30,
    wavelength=128,
    num_pmts=162,
    num_ticks=1000,
    max_time=100,
    photons_per_batch=200_000,
):
    from photonlib.meta import VoxelMeta

    g = build_detector_from_config(
        detector_config,
        flatten=True,
        include_wires=True,
        include_active=True,
        include_cathode=True,
        include_cavity=True,
    )
    sim = Simulation(g, geant4_processes=0, photon_tracking=0, particle_tracking=0)
    writer = H5Writer(output_filename, num_pmts=num_pmts, num_ticks=num_ticks, max_time=max_time)
    meta = VoxelMeta(shape=voxel_shape, ranges=voxel_ranges)

    for idx in range(voxel_index_start, voxel_index_start + batch_size):
        pos = meta.idx_to_coord(idx)
        phot = create_photon_bomb(nphotons, pos, voxel_size=voxel_size, wavelength=wavelength)
        for i, ev in enumerate(
            sim.simulate(
                phot,
                run_daq=False,
                photons_per_batch=photons_per_batch,
                keep_photons_beg=False,
                keep_photons_end=False,
                keep_hits=False,
                keep_flat_hits=True,
                max_steps=1000,
                verbose=True,
            )
        ):
            writer.save(ev, idx)
    writer.close()

def main():

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_filename", type=str, default="waveform_map.h5")
    parser.add_argument("--detector_config", type=str, default="detector_config_reflect_reflect3wires")
    parser.add_argument("--voxel_shape", type=str, default="77,144,144") # TODO: get from detector config
    parser.add_argument("--voxel_ranges", type=str, default="(-2310,0),(-2160,2160),(-2160,2160)") # TODO: get from detector config
    parser.add_argument("--voxel_index_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=720)
    parser.add_argument("--nphotons", type=int, default=200_000)
    parser.add_argument("--voxel_size", type=float, default=30)
    parser.add_argument("--wavelength", type=float, default=128)
    parser.add_argument("--num_pmts", type=int, default=162) # TODO: get from detector config
    parser.add_argument("--num_ticks", type=int, default=1000, help="number of ticks to save")
    parser.add_argument("--max_time", type=float, default=100, help="max time in ns")

    args = parser.parse_args()
    voxel_shape = tuple(map(int, args.voxel_shape.split(",")))
    voxel_ranges = tuple(map(float, args.voxel_ranges.split(",")))


    run_waveform_map(
        output_filename=args.output_filename,
        detector_config=args.detector_config,
        voxel_shape=voxel_shape,
        voxel_ranges=voxel_ranges,
        voxel_index_start=args.voxel_index_start,
        batch_size=args.batch_size,
        nphotons=args.nphotons,
        voxel_size=args.voxel_size,
        wavelength=args.wavelength,
        num_pmts=args.num_pmts,
        num_ticks=args.num_ticks,
        max_time=args.max_time,
        photons_per_batch=args.nphotons,
    )

if __name__ == "__main__":
    main()
