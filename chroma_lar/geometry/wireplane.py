import numpy as np
from math import cos, sin
import chroma.geometry as geometry
import chroma.make as make
import chroma.detector as detector

__exports__ = ["add_wires"]

def _clip_segment_to_rectangle(p0, d, y_half, z_half):
    """
    clip an infinite line  L(t) = p0 + d*t  (d is **unit**) to the rectangle
    |y|<=y_half, |z|<=z_half  in the Y-Z plane.
    Returns (t_min, t_max) or None if the line misses the rectangle.
    """
    t_low, t_high = -np.inf, np.inf

    #   y limits
    if abs(d[0]) < 1e-12:  # line is (almost) parallel to Z
        if abs(p0[0]) > y_half:  # outside band → no hit
            return None
    else:
        t1, t2 = (-y_half - p0[0]) / d[0], (y_half - p0[0]) / d[0]
        t_low, t_high = max(t_low, min(t1, t2)), min(t_high, max(t1, t2))

    #   z limits
    if abs(d[1]) < 1e-12:  # line is (almost) parallel to Y
        if abs(p0[1]) > z_half:
            return None
    else:
        t1, t2 = (-z_half - p0[1]) / d[1], (z_half - p0[1]) / d[1]
        t_low, t_high = max(t_low, min(t1, t2)), min(t_high, max(t1, t2))

    return (t_low, t_high) if t_low < t_high else None


def make_wire_plane(
    ly,
    lz,
    wire_pitch,  # distance between wires, mm
    wire_angle,  # radians, 0 = along Y (+horizontal), π/2 = along Z (vertical)
    wire_diameter,  # mm
    offset=0.0,  # shift of the whole bundle along its normal direction, mm
    nsteps=64,  # cylinder tessellation
):
    """
    Return a list of Mesh objects, one per wire, covering the rectangle
    |y|<=ly/2, |z|<=lz/2, oriented at `wire_angle`, with centre-to-centre pitch
    `wire_pitch` and diameter `wire_diameter`.
    """
    radius = 0.5 * wire_diameter
    y_half, z_half = 0.5 * ly, 0.5 * lz

    # unit direction vector of a wire (in the Y-Z plane)
    d = np.array([cos(wire_angle), sin(wire_angle)])  # (dy, dz)
    # unit normal (points “across” the wires, used to step pitches)
    n = np.array([-sin(wire_angle), cos(wire_angle)])

    # maximum distance from rectangle centre to cover all hits
    max_span = np.hypot(ly, lz) * 0.5 + wire_pitch
    # equally spaced intercepts along the normal
    intercepts = np.arange(-max_span + offset, max_span + wire_pitch, wire_pitch)

    meshes = []
    base_cyl = make.cylinder(radius, 1.0, nsteps=nsteps)  # unit-length cylinder along +Y

    # rotation that aligns +Y with direction d→
    rot_axis = np.array([sin(wire_angle), 0, 0])  # actually ±X axis
    rot_angle = wire_angle  # |angle| around X
    # pre-compute rotation matrix (Rodrigues)
    cA, sA = cos(rot_angle), sin(rot_angle)
    R = np.array([[1, 0, 0], [0, cA, -sA], [0, sA, cA]])

    for s in intercepts:
        # line centre point at closest approach to rectangle centre
        p0 = n * s  # (y0, z0)

        clip = _clip_segment_to_rectangle(p0, d, y_half, z_half)
        if clip is None:
            continue  # this wire misses the rectangle entirely

        t0, t1 = clip
        seg_len = t1 - t0  # in "t" units (=mm because |d|=1)
        seg_mid = p0 + d * (0.5 * (t0 + t1))  # (y, z) of segment centre

        # clone the unit cylinder, scale to correct length, rotate, translate
        v = base_cyl.vertices.copy()
        v[:, 1] *= seg_len  # stretch along local +Y (its axis)
        v = v @ R.T  # rotate into (d) direction
        v[:, 1] += seg_mid[0]  # Y translation
        v[:, 2] += seg_mid[1]  # Z translation
        # (X remains 0 – on the Y-Z plane)

        meshes.append(
            geometry.Mesh(v, base_cyl.triangles, remove_duplicate_vertices=False)
        )

    return meshes


def add_wires(
    g: detector.Detector,
    ly: float,
    lz: float,
    active_dimensions: dict,
    wire_diameter: float,
    wire_pitch: float,
    wire_angles: list,
    wire_offsets: list,
    wire_nsteps: int,
    inner_material: geometry.Material = None,
    outer_material: geometry.Material = None,
    surface: geometry.Surface = None,
    default_optics: detector.Detector = None,
):
    # define colors for wire planes
    wire_colors = [0x000000FF, 0x0000FF00, 0x00FF0000]  # blue, green, red

    if inner_material is None:
        inner_material = default_optics.steel_material
    if outer_material is None:
        outer_material = default_optics.lar
    if surface is None:
        surface = default_optics.reflect99

    # loop through the three wire planes
    for i, (plane_angle, offset) in enumerate(zip(wire_angles, wire_offsets)):
        # generate wire meshes
        wire_meshes = make_wire_plane(
            ly=ly,
            lz=lz,
            wire_pitch=wire_pitch,
            wire_angle=plane_angle,
            wire_diameter=wire_diameter,
            offset=0.0,
            nsteps=wire_nsteps,
        )

        # add wires to detector
        for mesh in wire_meshes:
            # position at x = active boundary + offset
            x_pos_1 = active_dimensions["x"][1] + offset
            x_pos_0 = active_dimensions["x"][0] - offset

            mesh_solid = geometry.Solid(
                mesh,
                material1=inner_material,
                material2=outer_material,
                color=wire_colors[i],
                surface=surface,
            )

            # add wire at +x position
            g.add_solid(mesh_solid, displacement=(x_pos_0, 0, 0))

            # add wire at -x position
            g.add_solid(mesh_solid, displacement=(x_pos_1, 0, 0))
    return g