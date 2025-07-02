from .pmt import generate_pmt_positions
from .wireplane import add_wires
from .pmt import build_r5912_pmt

import numpy as np
import chroma.geometry as geometry
import chroma.make as make
import chroma.transform as transform
import chroma.detector as detector
from chroma.loader import create_geometry_from_obj
from typing import Literal

def in2mm(in_value):
    """Convert inches to millimeters."""
    return in_value * 25.4

def build_detector(
    # Geometry parameters
    active_dimensions={
        "x": [-2160.0, 2160.0],
        "y": [-2160.0, 2160.0],
        "z": [-2160.0, 2160.0],
    },
    cavity_scale=1.5,  # Cavity size relative to active volume
    # PMT parameters
    pmt_photocathode_surface=None,
    pmt_back_surface=None,
    pmt_glass_material=None,
    pmt_spacing=500,  # mm
    pmt_gap=10,  # mm from active volume boundary
    pmt_nsteps=20,  # Tessellation for PMT model
    pmt_diameter_in = 8, # in
    # Wire plane parameters
    wire_diameter=0.150,  # mm
    wire_pitch=3.0,  # mm
    wire_angles=[np.pi / 2, np.pi / 3, -np.pi / 3],  # U, V, Y planes
    wire_offsets=[0.0, -3.0, -6.0],  # mm, relative to active boundary
    wire_nsteps=32,  # Tessellation for wire cylinders
    wire_inner_material=None,
    wire_surface=None,
    # Cathode parameters
    cathode_thickness=6,  # mm
    cathode_inner_material=None,
    cathode_surface=None,
    # Materials
    default_optics=None,
    target_material=None,
    # Active volume parameters
    active_surface=None,
    # Other options
    include_cavity=True,
    include_active=True,
    include_pmts=True,
    include_wires=True,
    include_cathode=True,
    # BVH options
    flatten=True,
):
    """
    Build a complete detector with all components.

    Parameters
    ----------
    active_dimensions : dict
        Dictionary with x, y, z ranges for active volume
    cavity_scale : float
        Scale factor for cavity relative to active volume
    pmt_spacing : float
        Spacing between PMTs in mm
    pmt_gap : float
        Gap between active volume and PMT planes in mm
    pmt_nsteps : int
        Tessellation parameter for PMT model
    wire_diameter : float
        Diameter of wire in mm
    wire_pitch : float
        Spacing between wires in mm
    wire_angles : list
        List of angles (radians) for the three wire planes
    wire_offsets : list
        Offset of each wire plane from TPC boundary
    wire_nsteps : int
        Tessellation parameter for wire cylinders
    default_optics : object
        Optics database for materials
    active_surface : Surface, optional
        Surface for the active volume cube
    include_cavity : bool
        Whether to include cavity volume
    include_active : bool
        Whether to include active volume
    include_pmts : bool
        Whether to include PMTs
    include_wires : bool
        Whether to include wire planes
    include_cathode : bool
        Whether to include cathode

    Returns
    -------
    detector.Detector
        Fully constructed detector object
    """

    if target_material is None:
        target_material = default_optics.lar

    # Calculate dimensions from ranges
    lx = active_dimensions["x"][1] - active_dimensions["x"][0]
    ly = active_dimensions["y"][1] - active_dimensions["y"][0]
    lz = active_dimensions["z"][1] - active_dimensions["z"][0]

    g = detector.Detector(target_material)

    if include_cavity:
        cavity = make.box(
            cavity_scale * lx, cavity_scale * ly, cavity_scale * lz, center=(0, 0, 0)
        )
        cavity_solid = geometry.Solid(
            cavity,
            target_material,
            default_optics.vacuum,
            color=0xFFFFFFFF,
            surface=default_optics.reflect00,
        )
        g.add_solid(cavity_solid)

    # generate pmt positions
    pmt_positions, pmt_indices, pmt_directions = generate_pmt_positions(
        lx=lx,
        ly=ly,
        lz=lz,
        spacing_y=pmt_spacing,
        spacing_z=pmt_spacing,
        gap_pmt_active=pmt_gap,
    )

    # create pmt model
    pmt = build_r5912_pmt(
        glass_thickness=3,
        nzsteps=pmt_nsteps,
        nsteps=64,
        diameter=in2mm(pmt_diameter_in),
        outer_material=target_material,
        glass=pmt_glass_material,
        vacuum=default_optics.vacuum,
        photocathode_surface=pmt_photocathode_surface,
        back_surface=pmt_back_surface,
        default_optics=default_optics,
    )
    if include_pmts:
        # add pmt to detector
        for p, r, i in zip(pmt_positions, pmt_directions, pmt_indices):
            # convert positions and directions if they're torch tensors
            if hasattr(p, "numpy"):
                p = p.numpy()
            if hasattr(r, "numpy"):
                r = r.numpy()

            # add pmt to detector
            g.add_pmt(
                pmt,
                displacement=p,
                rotation=transform.gen_rot(np.array([0.0, -1.0, 0]), r),
                channel_type=i
            )

    if include_active:
        if active_surface is None:
            active_surface = default_optics.reflect99


        active = make.box(
            lx + abs(pmt.mesh.vertices[:, 1].min()) * 2, ly, lz, center=(0, 0, 0)
        )
        active_solid = geometry.Solid(
            mesh=active,
            material1=target_material,
            material2=default_optics.vacuum,
            surface=active_surface,
            color=0xA0A0A0A0
        )
        g.add_solid(active_solid)

    if include_cathode:
        if cathode_inner_material is None:
            cathode_inner_material = default_optics.steel_material
        if cathode_surface is None:
            cathode_surface = default_optics.reflect99
        
        add_cathode(
            g,
            ly,
            lz,
            cathode_thickness,
            cathode_inner_material,
            target_material,
            cathode_surface,
            default_optics
        )
    if include_wires:
        add_wires(
            g,
            ly,
            lz,
            active_dimensions,
            wire_diameter,
            wire_pitch,
            wire_angles,
            wire_offsets,
            wire_nsteps,
            inner_material=wire_inner_material,
            outer_material=target_material,
            surface=wire_surface,
            default_optics=default_optics
        )

    if flatten:
        return create_geometry_from_obj(g)

    return g

def add_cathode(
    g: detector.Detector,
    ly: float,
    lz: float,
    cathode_thickness: float,
    inner_material: geometry.Material = None,
    outer_material: geometry.Material = None,
    surface: geometry.Surface = None,
    default_optics: detector.Detector = None,
):
    
    if inner_material is None:
        inner_material = default_optics.steel_material
    if outer_material is None:
        outer_material = default_optics.lar
    if surface is None:
        surface = default_optics.reflect99

    cathode = make.box(cathode_thickness, ly, lz, center=(0, 0, 0))
    cathode_solid = geometry.Solid(
        cathode,
        material1=inner_material,
        material2=outer_material,
        surface=surface,
        color=0xA0A0A0A0
    )

    g.add_solid(cathode_solid)
