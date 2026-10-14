"""Write the 8 validation voxels (cathode, anode, walls, corners) of the 3 cm
reflect3wires LUT grid to the .npy file given as the only argument."""

import sys

import numpy as np
import torch
from photonlib.meta import VoxelMeta

RANGES = ((-2310, 0), (-2160, 2160), (-2160, 2160))
VOXEL_SIZE = 30
POINTS = [
    (-60, 0, 0), (-1155, 0, 0), (-2250, 0, 0),              # cathode, middle, anode
    (-1155, 2100, 0), (-1155, -2100, 0), (-1155, 0, 2100),  # walls
    (-2250, 2100, 2100), (-60, -2100, -2100),               # corners
]

shape = tuple((hi - lo) // VOXEL_SIZE for lo, hi in RANGES)
meta = VoxelMeta(shape=shape, ranges=RANGES)
ids = meta.coord_to_voxel(torch.tensor(POINTS, dtype=torch.float32)).numpy().astype(np.int32)
assert len(set(ids.tolist())) == len(POINTS), ids
np.save(sys.argv[1], ids)
print(ids)
