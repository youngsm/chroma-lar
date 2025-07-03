import os
import logging
import numpy as np
import h5py

from chroma.log import logger
logger.setLevel(logging.DEBUG)
from chroma.sim import Simulation
from chroma.event import Photons

from chroma_lar.geometry import build_detector_from_config
from photonlib.meta import VoxelMeta


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


def __configure__(db):
    """Modify fields in the database here"""
    db.output_filename = "waveform_map.h5"
    db.detector_config = "detector_config_reflect_reflect3wires"
    db.voxel_shape = (77, 144, 144)
    db.voxel_ranges = ((-2310, 0), (-2160, 2160), (-2160, 2160))
    db.voxel_index_start = 0
    db.batch_size = 720
    db.nphotons = 200_000
    db.voxel_size = 30
    db.wavelength = 128
    db.num_pmts = 162
    db.num_ticks = 1000
    db.max_time = 100
    
    db.chroma_g4_processes = 0
    db.chroma_photon_tracking = 0
    db.chroma_particle_tracking = 0
    db.chroma_daq = False
    db.chroma_photons_per_batch = db.nphotons
    db.chroma_keep_photons_beg = False
    db.chroma_keep_photons_end = False
    db.chroma_keep_hits = False
    db.chroma_keep_flat_hits = True
    db.chroma_max_steps = 1000


def __define_geometry__(db):
    """Returns a chroma Detector or Geometry"""
    geometry = build_detector_from_config(
        db.detector_config,
        flatten=True,
        include_wires=True,
        include_active=True,
        include_cathode=True,
        include_cavity=True,
    )
    return geometry


def __event_generator__(db):
    """A generator to yield chroma Events (or something a chroma Simulation can
    convert to a chroma Event)."""
    meta = VoxelMeta(shape=db.voxel_shape, ranges=db.voxel_ranges)
    db.meta = meta
    db.voxel_ids = range(db.voxel_index_start, db.voxel_index_start + db.batch_size)
    
    for idx in db.voxel_ids:
        pos = meta.idx_to_coord(idx)
        yield create_photon_bomb(db.nphotons, pos, voxel_size=db.voxel_size, wavelength=db.wavelength)


def __simulation_start__(db):
    """Called at the start of the event loop"""
    db.writer = H5Writer(
        db.output_filename, 
        num_pmts=db.num_pmts, 
        num_ticks=db.num_ticks, 
        max_time=db.max_time
    )
    db.current_voxel_idx = 0
    db.num_events = db.batch_size


def __process_event__(db, ev):
    """Called for each generated event"""
    voxel_id = db.voxel_ids[db.current_voxel_idx]
    db.writer.save(ev, voxel_id)
    db.current_voxel_idx += 1


def __simulation_end__(db):
    """Called at the end of the event loop"""
    db.writer.close() 