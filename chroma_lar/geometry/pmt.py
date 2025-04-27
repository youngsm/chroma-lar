import numpy as np
from math import pi
import chroma.tools as tools
import chroma.geometry as geometry
import chroma.make as make
import torch
import os
import numpy as np
import chroma.geometry as geometry

__exports__ = ["generate_pmt_positions", "build_r5912_pmt"]


def in2mm(in_value):
    return in_value * 25.4


def generate_pmt_positions(lx, ly, lz, spacing_y, spacing_z, gap_pmt_active):
    """
    Generate hexagonal grid of PMTs on the y-z plane of a cubic detector
    :param lx: length of the cube in x direction
    :param ly: length of the cube in y direction
    :param lz: length of the cube in z direction
    :param spacing_y: spacing between PMTs in y direction
    :param spacing_z: spacing between PMTs in z direction
    :param gap_pmt_active: spacing between a pmt plane and the closest active volume boundary
    :return:
    pmt_coords: coordinates of the PMTs
    pmt_directions: direction vectors for PMTs (facing inward)
    """

    grid_y = int(ly / spacing_y)
    grid_z = int(lz / spacing_z)
    print(f"Total PMT number is {2 * grid_y * grid_z}")

    # Generate hexagonal grid coordinates
    y_side, z_side = torch.meshgrid(torch.arange(grid_y), torch.arange(grid_z), indexing='ij')
    y_side = y_side * spacing_y - ly / 2
    z_side = z_side * spacing_z - lz / 2
    y_side = y_side.to(torch.float32)
    z_side = z_side.to(torch.float32)
    for i in range(y_side.shape[1]):
        if i % 2 == 1:
            y_side[:, i] += spacing_y / 2

    y = torch.tile(y_side, (2, 1, 1)).flatten()  # reshape(2, -1)
    z = torch.tile(z_side, (2, 1, 1)).flatten()  # reshape(2, -1)

    num_lo_pmt = num_hi_pmt = int(len(y) / 2)
    lo_x_value = -lx / 2 - gap_pmt_active
    hi_x_value = lx / 2 + gap_pmt_active
    x = torch.concat(
        (torch.full((num_lo_pmt,), lo_x_value), torch.full((num_hi_pmt,), hi_x_value))
    )
    
    # Create direction vectors - PMTs should face inward
    # For PMTs on the -x side (lo_x_value), direction is +x (1,0,0)
    # For PMTs on the +x side (hi_x_value), direction is -x (-1,0,0)
    directions = torch.zeros((len(x), 3), dtype=torch.float32)
    directions[:num_lo_pmt, 0] = 1.0  # -x side PMTs face +x direction
    directions[num_lo_pmt:, 0] = -1.0  # +x side PMTs face -x direction
    
    # pmt_coords = torch.column_stack((x.flatten(), y.flatten(), z.flatten()))
    # change to swap between the sides
    pmt_coords = torch.stack((x, y, z), dim=-1)
    return pmt_coords, torch.arange(pmt_coords.shape[0], dtype=torch.int32), directions


def split_pmt_profile(
    y_min=65,
    split_threshold=220,
    target_diameter=190,
    downsample_factor=10,
):
    """
    Process a PMT profile by splitting it into top and bottom parts based on diameter criteria.

    Args:
        profile_path (str): Path to the CSV file containing PMT contour data
        y_min (float): Minimum y-value to consider in the profile
        split_threshold (float): Minimum y-value for considering diameter matching
        target_diameter (float): Target diameter to match for splitting the profile
        downsample_factor (int): Factor by which to downsample the profile points

    Returns:
        tuple: (cap_profile, bottom_profile, downsampled_cap, downsampled_bottom, y_threshold, actual_diameter)
    """
    # Read the profile data
    # make the path relative to the current file
    profile_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pmt_contour_mm.csv")
    profile = tools.read_csv(profile_path)
    profile = profile[profile[:, 1] > y_min]

    # Calculate the diameter at each y position
    unique_y_values = np.unique(profile[:, 1])
    diameters = []
    y_values = []

    for y in unique_y_values:
        points_at_y = profile[profile[:, 1] == y]
        if len(points_at_y) >= 2:  # Need at least 2 points to calculate diameter
            diameter = np.max(points_at_y[:, 0]) - np.min(points_at_y[:, 0])
            diameters.append(diameter)
            y_values.append(y)

    diameters = np.array(diameters)
    y_values = np.array(y_values)

    # Find the y-value where diameter is closest to target_diameter and y > split_threshold
    valid_indices = y_values > split_threshold
    if np.any(valid_indices):
        valid_diameters = diameters[valid_indices]
        valid_y_values = y_values[valid_indices]
        closest_idx = np.argmin(np.abs(valid_diameters - target_diameter))
        y_threshold = valid_y_values[closest_idx]
        actual_diameter = valid_diameters[closest_idx]
    else:
        # Fallback if no valid points found
        y_threshold = split_threshold
        actual_diameter = 0

    # Get midpoint in X
    x_midpoint = (np.max(profile[:, 0]) + np.min(profile[:, 0])) / 2
    # Get top-most point in Y
    y_max = np.max(profile[:, 1])

    # Shift the coordinates to set the zero points
    profile[:, 0] -= x_midpoint  # X=0 at the midpoint
    profile[:, 1] = profile[:, 1] - y_max  # Y=0 at the top-most point
    y_threshold = y_threshold - y_max

    return (
        profile,
        y_threshold,
    )


def build_r5912_pmt(
    glass_thickness=3.0,
    nzsteps=40,
    nsteps=64,
    diameter=in2mm(8),
    outer_material=None,
    glass=None,
    vacuum=None,
    photocathode_surface=None,
    back_surface=None,
    default_optics=None,
    return_individual_solids=False,
):

    """ returns R5912 PMT solid oriented along [0,1,0]"""
    profile, y_threshold = split_pmt_profile(downsample_factor=4)

    if outer_material is None:
        outer_material = default_optics.lar
    if glass is None:
        glass = default_optics.glass
    if vacuum is None:
        vacuum = default_optics.vacuum
    if photocathode_surface is None:
        photocathode_surface = default_optics.r5912_mod_photocathode
    if back_surface is None:
        back_surface = default_optics.glossy_surface

    # slice profile in half
    profile = profile[profile[:, 0] < 0]
    profile[:, 0] = -profile[:, 0]
    # order profile from base to face
    profile = profile[np.argsort(profile[:, 1])] * diameter / (in2mm(8))
    # set x coordinate to 0.0 for first and last profile along the profile

    mask = profile[:, 1] > -150 * diameter / (in2mm(8))
    idx = np.argwhere(mask)[0, 0]
    profile[0, 0] = 0.0  # close it
    profile[-1, 0] = 0.0  # close it

    back_end = np.concatenate([profile[[0]], [[profile[idx, 0], profile[0, 1]]]])
    front_end = profile[mask]

    r = np.sqrt(np.sum(np.square(front_end), axis=1))
    theta = np.arctan2(front_end[:, 1], front_end[:, 0])

    new_theta = np.linspace(theta[0], theta[-1], nzsteps)
    if pi / 2 not in new_theta:
        new_theta = np.sort(np.append(new_theta, pi / 2))
    new_r = np.interp(new_theta, theta, r)
    front_end = np.asarray([new_r * np.cos(new_theta), new_r * np.sin(new_theta)]).T
    front_end[-1, 0] = 0.0  # close it
    profile = np.concatenate([back_end, front_end])

    offset_profile = tools.offset(profile, -glass_thickness * (diameter / in2mm(8)))
    offset_profile[0, 0] = 0.0  # close it
    offset_profile[-1, 0] = 0.0  # close it

    outer_envelope_mesh = make.rotate_extrude(profile[:, 0], profile[:, 1], nsteps)
    inner_envelope_mesh = make.rotate_extrude(
        offset_profile[:, 0], offset_profile[:, 1], nsteps
    )

    outer_envelope = geometry.Solid(
        outer_envelope_mesh, glass, outer_material, color=0xD0E0E0E0
    )

    # photocathode starts at y=y_threshold
    photocathode = np.mean(inner_envelope_mesh.assemble(), axis=1)[:, 1] > y_threshold
    # photocathode = np.ones_like(photocathode).astype(bool)

    inner_envelope = geometry.Solid(
        inner_envelope_mesh,
        vacuum,
        glass,
        surface=np.where(photocathode, photocathode_surface, back_surface),
        color=np.where(photocathode, 0xC0FF9900, 0xD0E0E0E0),
    )

    if return_individual_solids:
        return outer_envelope, inner_envelope

    pmt = outer_envelope + inner_envelope
    return pmt
