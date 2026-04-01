"""Air-drawing engine: tracks fingertip path → voxel trail."""
import numpy as np
from collections import deque
from scipy.interpolate import splprep, splev
from utils.math_helpers import world_to_voxel
from utils.constants import VOXEL_SIZE


class PlaneMode:
    FREE = "free"
    X    = "x"
    Y    = "y"
    Z    = "z"


class DrawMode:
    FREEHAND = "freehand"
    STRAIGHT = "straight"
    BEZIER   = "bezier"


class AirDrawEngine:
    def __init__(self, voxel_world, current_type_fn, current_color_fn):
        self.world = voxel_world
        self.get_type  = current_type_fn
        self.get_color = current_color_fn

        self.active      = False
        self.mirror      = False
        self.plane_mode  = PlaneMode.FREE
        self.draw_mode   = DrawMode.FREEHAND
        self.erase_mode  = False

        self._trail: deque[np.ndarray] = deque(maxlen=500)
        self._preview_voxels: list[tuple] = []
        self._confirmed      = False
        self._min_step       = VOXEL_SIZE * 0.8

    # ── State ─────────────────────────────────
    def start(self):
        self.active = True
        self._trail.clear()
        self._preview_voxels.clear()
        self._confirmed = False

    def stop(self):
        self.active = False

    def confirm(self):
        """Place all preview voxels into the world."""
        for coord in self._preview_voxels:
            if self.erase_mode:
                self.world.remove_voxel(coord)
            else:
                self.world.set_voxel(coord, self.get_type(), self.get_color())
                if self.mirror:
                    mx, my, mz = coord
                    self.world.set_voxel((-mx, my, mz), self.get_type(), self.get_color())
        self._preview_voxels.clear()
        self.active = False

    # ── Tick ──────────────────────────────────
    def update(self, world_pos: np.ndarray, speed: float = 1.0):
        if not self.active:
            return

        pos = self._apply_plane_lock(world_pos)

        if self._trail:
            last = self._trail[-1]
            if np.linalg.norm(pos - last) < self._min_step / max(speed, 0.1):
                return
        self._trail.append(pos.copy())

        if self.draw_mode == DrawMode.FREEHAND:
            self._preview_voxels = self._trail_to_voxels(list(self._trail))
        elif self.draw_mode == DrawMode.STRAIGHT:
            if len(self._trail) >= 2:
                self._preview_voxels = self._line_voxels(
                    self._trail[0], self._trail[-1])
        elif self.draw_mode == DrawMode.BEZIER:
            self._preview_voxels = self._bezier_voxels(list(self._trail))

    def _apply_plane_lock(self, pos: np.ndarray) -> np.ndarray:
        if not self._trail or self.plane_mode == PlaneMode.FREE:
            return pos
        anchor = self._trail[0]
        p = pos.copy()
        if self.plane_mode == PlaneMode.X:
            p[0] = anchor[0]
        elif self.plane_mode == PlaneMode.Y:
            p[1] = anchor[1]
        elif self.plane_mode == PlaneMode.Z:
            p[2] = anchor[2]
        return p

    def _trail_to_voxels(self, pts: list) -> list:
        coords = []
        for p in pts:
            c = world_to_voxel(p)
            if c not in coords:
                coords.append(c)
        return coords

    def _line_voxels(self, a: np.ndarray, b: np.ndarray) -> list:
        steps = int(np.linalg.norm(b - a) / VOXEL_SIZE) + 1
        coords = []
        for i in range(steps + 1):
            t = i / max(steps, 1)
            p = a + (b - a) * t
            c = world_to_voxel(p)
            if c not in coords:
                coords.append(c)
        return coords

    def _bezier_voxels(self, pts: list) -> list:
        if len(pts) < 4:
            return self._trail_to_voxels(pts)
        arr = np.array(pts).T
        try:
            tck, _ = splprep(arr, s=0, k=min(3, len(pts)-1))
            t_new = np.linspace(0, 1, len(pts) * 4)
            interp = np.array(splev(t_new, tck)).T
            return self._trail_to_voxels(list(interp))
        except Exception:
            return self._trail_to_voxels(pts)

    @property
    def preview_voxels(self):
        return self._preview_voxels

    @property
    def trail_screen_pts(self) -> list:
        return list(self._trail)
