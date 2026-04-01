"""Single voxel data class."""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Voxel:
    vtype: int                      # index into VOXEL_TYPES
    color: Tuple[int, int, int, int]  # RGBA tint override (0-255)

    def is_air(self) -> bool:
        return self.vtype == 0
