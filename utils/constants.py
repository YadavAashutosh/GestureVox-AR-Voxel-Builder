"""Global constants for the AR Voxel Builder."""
import numpy as np

# ── Window ──────────────────────────────────
WINDOW_W = 1280
WINDOW_H = 720
TARGET_FPS_WEBCAM = 30
TARGET_FPS_RENDER = 60

# ── Camera / AR ─────────────────────────────
FOCAL_LENGTH_FACTOR = 1.2          # fx = FOCAL_LENGTH_FACTOR * width
NEAR_PLANE = 0.1
FAR_PLANE  = 500.0
FOV_Y      = 60.0                  # degrees

# Palm size reference (pixels) at ~40 cm distance
PALM_REF_PX   = 120.0
PALM_REF_DEPTH = 40.0              # cm

# ── Voxel World ──────────────────────────────
VOXEL_SIZE    = 2.0                # world-units per voxel edge
GRID_ORIGIN   = np.array([0.0, 0.0, -80.0], dtype=np.float32)
CHUNK_SIZE    = 8                  # voxels per chunk axis
UNDO_MAX      = 100

# ── Gesture ──────────────────────────────────
GESTURE_CONFIDENCE  = 0.80
GESTURE_COOLDOWN_S  = 0.35         # seconds between repeated triggers
PINCH_DIST_THRESH   = 0.055        # normalised distance (MediaPipe coords)
PINCH_HOLD_S        = 2.0
SWIPE_DIST_THRESH   = 0.15
DOUBLE_PINCH_GAP_S  = 0.45

# ── Voxel Types ──────────────────────────────
VOXEL_TYPES = [
    ("Air",         (  0,   0,   0, 0  )),
    ("Stone",       (128, 128, 128, 255)),
    ("Grass",       ( 76, 153,   0, 255)),
    ("Dirt",        (139,  90,  43, 255)),
    ("Wood",        (160, 110,  55, 255)),
    ("Leaves",      ( 34, 139,  34, 255)),
    ("Sand",        (237, 201, 140, 255)),
    ("Water",       ( 64, 164, 223, 200)),
    ("Lava",        (207,  56,   2, 255)),
    ("Snow",        (240, 240, 255, 255)),
    ("Ice",         (173, 216, 230, 200)),
    ("Gold",        (255, 215,   0, 255)),
    ("Diamond",     (185, 242, 255, 255)),
    ("Brick",       (178,  34,  34, 255)),
    ("Glass",       (200, 230, 255, 100)),
    ("Obsidian",    ( 30,  10,  50, 255)),
    ("Neon Red",    (255,  20,  60, 255)),
    ("Neon Green",  ( 20, 255,  80, 255)),
    ("Neon Blue",   ( 20,  80, 255, 255)),
    ("Neon Yellow", (255, 240,   0, 255)),
    ("Cloud",       (255, 255, 255, 180)),
    ("Bedrock",     ( 50,  50,  50, 255)),
]

VOXEL_COLORS_GL = {
    i: (r/255, g/255, b/255, a/255)
    for i, (_, (r, g, b, a)) in enumerate(VOXEL_TYPES)
}

# ── Colors ───────────────────────────────────
HUD_COLOR       = (0, 255, 170)        # OpenCV BGR-ish (we store RGB)
GHOST_COLOR     = (0.3, 0.5, 1.0, 0.35)
SKELETON_COLORS = [
    (255,  80,  80),  # thumb
    (255, 180,  60),  # index
    ( 80, 255, 120),  # middle
    ( 80, 180, 255),  # ring
    (200,  80, 255),  # pinky
]
TRAIL_COLOR     = (0, 220, 255)        # BGR
