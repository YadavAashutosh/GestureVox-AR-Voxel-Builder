"""VoxelWorld with chunk-based indexing and undo/redo."""
import copy
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
from voxel_core.voxel import Voxel
from utils.constants import CHUNK_SIZE, UNDO_MAX, VOXEL_TYPES


Coord = Tuple[int, int, int]


def _chunk_key(x, y, z):
    return (x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE)


class VoxelWorld:
    def __init__(self):
        self.voxels: Dict[Coord, Tuple[int, Tuple]] = {}   # coord → (type, color)
        self._chunks: Dict[Tuple, set] = defaultdict(set)  # chunk_key → set of coords
        self._undo_stack: deque = deque(maxlen=UNDO_MAX)
        self._redo_stack: deque = deque(maxlen=UNDO_MAX)
        self._dirty = True  # renderer should re-upload

    # ── Mutation ─────────────────────────────
    def set_voxel(self, coord: Coord, vtype: int, color: tuple | None = None):
        if color is None:
            color = VOXEL_TYPES[vtype][1]
        self._push_undo(coord)
        self.voxels[coord] = (vtype, color)
        self._chunks[_chunk_key(*coord)].add(coord)
        self._dirty = True

    def remove_voxel(self, coord: Coord):
        if coord in self.voxels:
            self._push_undo(coord)
            del self.voxels[coord]
            self._chunks[_chunk_key(*coord)].discard(coord)
            self._dirty = True

    def get_voxel(self, coord: Coord) -> Optional[Tuple[int, tuple]]:
        return self.voxels.get(coord)

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.voxels.clear()
        self._chunks.clear()
        self._dirty = True

    # ── Undo / Redo ──────────────────────────
    def _push_undo(self, coord: Coord):
        prev = self.voxels.get(coord)
        self._undo_stack.append((coord, prev))
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        coord, prev = self._undo_stack.pop()
        cur = self.voxels.get(coord)
        self._redo_stack.append((coord, cur))
        if prev is None:
            self.voxels.pop(coord, None)
            self._chunks[_chunk_key(*coord)].discard(coord)
        else:
            self.voxels[coord] = prev
            self._chunks[_chunk_key(*coord)].add(coord)
        self._dirty = True

    def redo(self):
        if not self._redo_stack:
            return
        coord, nxt = self._redo_stack.pop()
        cur = self.voxels.get(coord)
        self._undo_stack.append((coord, cur))
        if nxt is None:
            self.voxels.pop(coord, None)
            self._chunks[_chunk_key(*coord)].discard(coord)
        else:
            self.voxels[coord] = nxt
            self._chunks[_chunk_key(*coord)].add(coord)
        self._dirty = True

    # ── Flood fill ───────────────────────────
    def flood_fill(self, start: Coord, new_type: int, new_color: tuple):
        if start not in self.voxels:
            return
        target_type, _ = self.voxels[start]
        if target_type == new_type:
            return
        visited = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c in visited or c not in self.voxels:
                continue
            vt, _ = self.voxels[c]
            if vt != target_type:
                continue
            visited.add(c)
            x, y, z = c
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nb = (x+dx, y+dy, z+dz)
                if nb not in visited:
                    stack.append(nb)
        for c in visited:
            self.set_voxel(c, new_type, new_color)

    # ── Chunk queries ────────────────────────
    def get_chunk_coords(self, chunk_key) -> set:
        return self._chunks.get(chunk_key, set())

    def all_chunks(self):
        return list(self._chunks.keys())

    # ── Box select ───────────────────────────
    def get_region(self, c1: Coord, c2: Coord) -> Dict[Coord, Tuple]:
        x0, x1 = sorted([c1[0], c2[0]])
        y0, y1 = sorted([c1[1], c2[1]])
        z0, z1 = sorted([c1[2], c2[2]])
        return {
            c: v for c, v in self.voxels.items()
            if x0 <= c[0] <= x1 and y0 <= c[1] <= y1 and z0 <= c[2] <= z1
        }

    def paste_region(self, region: Dict[Coord, Tuple], offset: Coord):
        ox, oy, oz = offset
        for (x, y, z), (vt, col) in region.items():
            self.set_voxel((x + ox, y + oy, z + oz), vt, col)
