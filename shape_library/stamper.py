"""Stamps a procedural shape into the voxel world."""
import numpy as np
from shape_library.shapes import SHAPE_REGISTRY
from utils.math_helpers import world_to_voxel
from utils.constants import VOXEL_SIZE


class ShapeStamper:
    def __init__(self, voxel_world):
        self.world = voxel_world

    def stamp(self, shape_name: str, center_world: np.ndarray,
              vtype: int, color: tuple):
        fn = SHAPE_REGISTRY.get(shape_name)
        if fn is None:
            return
        offsets = fn()
        cx, cy, cz = world_to_voxel(center_world)
        for dx, dy, dz in offsets:
            self.world.set_voxel((cx+dx, cy+dy, cz+dz), vtype, color)
