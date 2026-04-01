"""23-gesture classifier using geometric rules on MediaPipe landmarks."""
import time
import numpy as np
from enum import Enum, auto
from utils.constants import (
    PINCH_DIST_THRESH, GESTURE_COOLDOWN_S,
    SWIPE_DIST_THRESH, PINCH_HOLD_S, DOUBLE_PINCH_GAP_S
)


class Gesture(Enum):
    NONE            = auto()
    PINCH           = auto()
    DOUBLE_PINCH    = auto()
    FIST            = auto()
    OPEN_PALM       = auto()
    POINT           = auto()
    TWO_FINGER_SCROLL_UP   = auto()
    TWO_FINGER_SCROLL_DOWN = auto()
    THREE_SWIPE_LEFT  = auto()
    THREE_SWIPE_RIGHT = auto()
    FOUR_FINGER_TAP   = auto()
    PINKY_RAISE       = auto()
    WRIST_ROTATE_LEFT  = auto()
    WRIST_ROTATE_RIGHT = auto()
    TWO_HAND_SPREAD   = auto()
    TWO_HAND_PINCH    = auto()
    BOTH_FISTS        = auto()
    L_SHAPE           = auto()
    OK_GESTURE        = auto()
    PEACE             = auto()
    THUMB_UP          = auto()
    THUMB_DOWN        = auto()
    ROCK_SIGN         = auto()
    HAND_TILT_LEFT    = auto()
    HAND_TILT_RIGHT   = auto()
    SLOW_PINCH_HOLD   = auto()
    DRAW_MODE         = auto()   # index only raised


def _dist(lms, a, b):
    ax, ay = lms[a].x, lms[a].y
    bx, by = lms[b].x, lms[b].y
    return float(np.hypot(ax - bx, ay - by))


def _fingers_up(lms):
    """Return list of booleans [thumb, index, middle, ring, pinky]."""
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    up = []
    # Thumb: compare x if hand is right-ish
    thumb_up = lms[4].x < lms[3].x  # mirrored webcam → right hand
    up.append(thumb_up)
    for tip, pip in zip(tips[1:], pips[1:]):
        up.append(lms[tip].y < lms[pip].y)
    return up


def _palm_tilt(lms) -> float:
    """Return wrist tilt angle in degrees (-90..90)."""
    wx, wy = lms[0].x, lms[0].y
    mx, my = lms[9].x, lms[9].y
    return float(np.degrees(np.arctan2(my - wy, mx - wx)))


class GestureClassifier:
    def __init__(self):
        self._last_gesture = Gesture.NONE
        self._last_time: dict[Gesture, float] = {}
        self._pinch_start: float | None = None
        self._last_pinch_time: float | None = None
        self._prev_wrist_x: float | None = None
        self._prev_scroll_y: float | None = None
        self._prev_palm_x: float | None = None
        self._history: list[tuple[float, np.ndarray]] = []  # (t, wrist)

    def classify(self, lms_list: list, delta_t: float = 0.033) -> Gesture:
        """
        lms_list: list of (landmarks, label) from HandTracker.get_all_hands()
        Returns dominant Gesture.
        """
        now = time.time()
        if not lms_list:
            self._pinch_start = None
            return Gesture.NONE

        lms, label = lms_list[0]

        # ── Two-hand gestures ────────────────
        if len(lms_list) >= 2:
            lms2, _ = lms_list[1]
            g = self._two_hand_gesture(lms, lms2, now)
            if g != Gesture.NONE:
                return self._apply_cooldown(g, now)

        fu = _fingers_up(lms)
        thumb, index, middle, ring, pinky = fu
        n_up = sum(fu)

        # ── Pinch ────────────────────────────
        pinch_d = _dist(lms, 4, 8)
        is_pinch = pinch_d < PINCH_DIST_THRESH

        if is_pinch:
            if self._pinch_start is None:
                self._pinch_start = now
            held = now - self._pinch_start
            # slow pinch hold
            if held >= PINCH_HOLD_S:
                self._pinch_start = None
                return self._apply_cooldown(Gesture.SLOW_PINCH_HOLD, now)
            # double pinch
            if self._last_pinch_time and (now - self._last_pinch_time) < DOUBLE_PINCH_GAP_S:
                self._last_pinch_time = None
                return self._apply_cooldown(Gesture.DOUBLE_PINCH, now)
        else:
            if self._pinch_start is not None:
                self._last_pinch_time = now
            self._pinch_start = None

        if is_pinch and not middle and not ring and not pinky:
            return self._apply_cooldown(Gesture.PINCH, now)

        # ── Fist ─────────────────────────────
        if n_up == 0:
            return self._apply_cooldown(Gesture.FIST, now)

        # ── Open palm ────────────────────────
        if n_up == 5:
            return self._apply_cooldown(Gesture.OPEN_PALM, now)

        # ── Draw mode (only index up) ─────────
        if index and not middle and not ring and not pinky and not thumb:
            return self._apply_cooldown(Gesture.DRAW_MODE, now)

        # ── Peace (index + middle) ────────────
        if index and middle and not ring and not pinky:
            # Check scroll vs peace by wrist movement
            wy = lms[0].y
            if self._prev_scroll_y is not None:
                dy = self._prev_scroll_y - wy
                if dy > SWIPE_DIST_THRESH * 0.5:
                    self._prev_scroll_y = wy
                    return self._apply_cooldown(Gesture.TWO_FINGER_SCROLL_UP, now)
                elif dy < -SWIPE_DIST_THRESH * 0.5:
                    self._prev_scroll_y = wy
                    return self._apply_cooldown(Gesture.TWO_FINGER_SCROLL_DOWN, now)
            self._prev_scroll_y = wy
            return self._apply_cooldown(Gesture.PEACE, now)

        # ── Three fingers: swipe ──────────────
        if index and middle and ring and not pinky:
            wx = lms[0].x
            if self._prev_wrist_x is not None:
                dx = wx - self._prev_wrist_x
                if dx > SWIPE_DIST_THRESH:
                    self._prev_wrist_x = wx
                    return self._apply_cooldown(Gesture.THREE_SWIPE_RIGHT, now)
                elif dx < -SWIPE_DIST_THRESH:
                    self._prev_wrist_x = wx
                    return self._apply_cooldown(Gesture.THREE_SWIPE_LEFT, now)
            self._prev_wrist_x = wx
            return Gesture.NONE

        # ── Four fingers ─────────────────────
        if index and middle and ring and pinky and not thumb:
            return self._apply_cooldown(Gesture.FOUR_FINGER_TAP, now)

        # ── Pinky raise ──────────────────────
        if pinky and not index and not middle and not ring:
            return self._apply_cooldown(Gesture.PINKY_RAISE, now)

        # ── Thumb up / down ──────────────────
        if thumb and not index and not middle and not ring and not pinky:
            tip_y = lms[4].y
            wrist_y = lms[0].y
            if tip_y < wrist_y - 0.1:
                return self._apply_cooldown(Gesture.THUMB_UP, now)
            elif tip_y > wrist_y + 0.05:
                return self._apply_cooldown(Gesture.THUMB_DOWN, now)

        # ── Rock sign (index + pinky) ─────────
        if index and pinky and not middle and not ring:
            return self._apply_cooldown(Gesture.ROCK_SIGN, now)

        # ── L-shape (thumb + index at 90°) ───
        if thumb and index and not middle and not ring and not pinky:
            return self._apply_cooldown(Gesture.L_SHAPE, now)

        # ── OK gesture (pinch all others up) ──
        if is_pinch and middle and ring and pinky:
            return self._apply_cooldown(Gesture.OK_GESTURE, now)

        # ── Wrist rotate ─────────────────────
        tilt = _palm_tilt(lms)
        if self._prev_wrist_x is not None:
            if tilt < -40:
                return self._apply_cooldown(Gesture.WRIST_ROTATE_LEFT, now)
            elif tilt > 40:
                return self._apply_cooldown(Gesture.WRIST_ROTATE_RIGHT, now)

        # ── Hand tilt ────────────────────────
        if index and not middle:
            palm_x = lms[0].x
            if self._prev_palm_x is not None:
                dx = palm_x - self._prev_palm_x
                if dx > SWIPE_DIST_THRESH * 0.6:
                    self._prev_palm_x = palm_x
                    return self._apply_cooldown(Gesture.HAND_TILT_RIGHT, now)
                elif dx < -SWIPE_DIST_THRESH * 0.6:
                    self._prev_palm_x = palm_x
                    return self._apply_cooldown(Gesture.HAND_TILT_LEFT, now)
            self._prev_palm_x = palm_x

        return Gesture.NONE

    def _two_hand_gesture(self, lms1, lms2, now) -> "Gesture":
        fu1 = _fingers_up(lms1)
        fu2 = _fingers_up(lms2)
        n1, n2 = sum(fu1), sum(fu2)

        # Spread / pinch: distance between index tips
        d = float(np.hypot(lms1[8].x - lms2[8].x, lms1[8].y - lms2[8].y))
        if d > 0.55:
            return Gesture.TWO_HAND_SPREAD
        if d < 0.10:
            return Gesture.TWO_HAND_PINCH

        if n1 == 0 and n2 == 0:
            return Gesture.BOTH_FISTS
        return Gesture.NONE

    def _apply_cooldown(self, g: Gesture, now: float) -> Gesture:
        last = self._last_time.get(g, 0.0)
        if now - last < GESTURE_COOLDOWN_S:
            return g  # still report gesture but caller decides debounce
        self._last_time[g] = now
        return g

    def get_fingertip_3d(self, lms, depth_cm: float) -> np.ndarray:
        """Return index fingertip position in camera space (cm)."""
        from utils.math_helpers import get_camera_matrix
        cam = get_camera_matrix()
        sx = int(lms[8].x * 1280)
        sy = int(lms[8].y * 720)
        fx, fy = cam[0, 0], cam[1, 1]
        cx, cy = cam[0, 2], cam[1, 2]
        z = depth_cm
        x = (sx - cx) * z / fx
        y = (sy - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)
