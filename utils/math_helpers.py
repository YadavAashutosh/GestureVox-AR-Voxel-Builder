"""Vector / matrix utilities shared across all modules."""
import numpy as np
from utils.constants import (
    WINDOW_W, WINDOW_H, FOCAL_LENGTH_FACTOR,
    VOXEL_SIZE, GRID_ORIGIN, PALM_REF_PX, PALM_REF_DEPTH
)


# ── Camera intrinsics (estimated) ─────────────
def get_camera_matrix() -> np.ndarray:
    fx = FOCAL_LENGTH_FACTOR * WINDOW_W
    fy = fx
    cx = WINDOW_W / 2
    cy = WINDOW_H / 2
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def project_3d_to_2d(pt3d: np.ndarray, cam_mat: np.ndarray,
                      view_mat: np.ndarray) -> tuple[int, int] | None:
    """Project a 3-D world point to 2-D screen coords."""
    p = np.append(pt3d, 1.0)
    cam_pt = view_mat @ p
    if cam_pt[2] <= 0:
        return None
    px = cam_mat[0, 0] * cam_pt[0] / cam_pt[2] + cam_mat[0, 2]
    py = cam_mat[1, 1] * cam_pt[1] / cam_pt[2] + cam_mat[1, 2]
    return int(px), int(py)


def unproject_2d_to_ray(sx: int, sy: int,
                         cam_mat: np.ndarray) -> np.ndarray:
    """Return normalised camera-space ray direction for screen pixel."""
    fx, fy = cam_mat[0, 0], cam_mat[1, 1]
    cx, cy = cam_mat[0, 2], cam_mat[1, 2]
    d = np.array([(sx - cx) / fx, (sy - cy) / fy, 1.0], dtype=np.float64)
    return d / np.linalg.norm(d)


# ── Depth from palm size ──────────────────────
def estimate_depth_from_palm(lms) -> float:
    """Return estimated palm depth in cm from MediaPipe landmarks."""
    wrist = np.array([lms[0].x * WINDOW_W, lms[0].y * WINDOW_H])
    mid   = np.array([lms[9].x * WINDOW_W, lms[9].y * WINDOW_H])
    px    = float(np.linalg.norm(mid - wrist))
    if px < 1e-3:
        return PALM_REF_DEPTH
    return PALM_REF_DEPTH * PALM_REF_PX / px


# ── Voxel ↔ world ────────────────────────────
def world_to_voxel(world: np.ndarray) -> tuple[int, int, int]:
    rel = (world - GRID_ORIGIN) / VOXEL_SIZE
    return int(round(rel[0])), int(round(rel[1])), int(round(rel[2]))


def voxel_to_world(vx: int, vy: int, vz: int) -> np.ndarray:
    return GRID_ORIGIN + np.array([vx, vy, vz], dtype=np.float32) * VOXEL_SIZE


# ── Misc ─────────────────────────────────────
def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rot_matrix_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], dtype=np.float32)


def rot_matrix_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0,  0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]], dtype=np.float32)


def translation_matrix(tx, ty, tz) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = tx, ty, tz
    return m


def perspective_matrix(fovy_deg, aspect, near, far) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    nf = 1 / (near - far)
    return np.array([
        [f/aspect, 0,               0,  0],
        [0,        f,               0,  0],
        [0,        0, (far+near)*nf, 2*far*near*nf],
        [0,        0,              -1,  0],
    ], dtype=np.float32)
