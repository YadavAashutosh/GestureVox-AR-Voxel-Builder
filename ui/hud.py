"""OpenCV HUD overlays: gesture text, voxel info, mode indicators."""
import cv2
import numpy as np
import time
from utils.constants import WINDOW_W, WINDOW_H, VOXEL_TYPES


FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX
COLOR_MAIN = (0, 255, 170)   # BGR neon green
COLOR_ACC  = (255, 180, 0)   # orange
COLOR_RED  = (60, 60, 255)
COLOR_WHITE= (255, 255, 255)
COLOR_DARK = (10, 10, 10)


class HUDOverlay:
    def __init__(self):
        self._gesture_text  = ""
        self._gesture_time  = 0.0
        self._gesture_dur   = 1.5
        self._info_lines: list[str] = []
        self._trail_pts: list[tuple] = []

    # ── Gesture flash ─────────────────────────
    def set_gesture(self, text: str):
        self._gesture_text = text
        self._gesture_time = time.time()

    # ── Draw everything ───────────────────────
    def draw(self, frame: np.ndarray, state: dict) -> np.ndarray:
        frame = frame.copy()
        frame = self._draw_top_bar(frame, state)
        frame = self._draw_bottom_bar(frame, state)
        frame = self._draw_gesture_popup(frame)
        frame = self._draw_crosshair(frame, state)
        frame = self._draw_trail(frame, state)
        frame = self._draw_depth_indicator(frame, state)
        return frame

    def _draw_top_bar(self, frame, state):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_W, 44), COLOR_DARK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        vt_idx = state.get("voxel_type", 1)
        vt_name = VOXEL_TYPES[vt_idx][0] if vt_idx < len(VOXEL_TYPES) else "?"
        vt_color_raw = VOXEL_TYPES[vt_idx][1][:3]
        vt_color_bgr = (vt_color_raw[2], vt_color_raw[1], vt_color_raw[0])

        cv2.putText(frame, f"VOXEL: {vt_name}", (12, 29),
                    FONT_BOLD, 0.7, COLOR_MAIN, 2, cv2.LINE_AA)
        cv2.circle(frame, (210, 22), 12, vt_color_bgr, -1)
        cv2.circle(frame, (210, 22), 12, COLOR_WHITE, 1)

        mode = state.get("mode", "PLACE")
        cv2.putText(frame, f"MODE: {mode}", (240, 29),
                    FONT, 0.6, COLOR_ACC, 1, cv2.LINE_AA)

        n_voxels = state.get("n_voxels", 0)
        cv2.putText(frame, f"VOXELS: {n_voxels}", (WINDOW_W - 180, 29),
                    FONT, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

        lighting = state.get("lighting", True)
        li_col = COLOR_MAIN if lighting else COLOR_RED
        cv2.putText(frame, "LIGHT", (WINDOW_W - 60, 29),
                    FONT, 0.5, li_col, 1, cv2.LINE_AA)
        return frame

    def _draw_bottom_bar(self, frame, state):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, WINDOW_H - 38), (WINDOW_W, WINDOW_H),
                      COLOR_DARK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        hints = [
            "PINCH=Place", "2xPINEH=Del", "PALM=Menu",
            "3→=Redo", "3←=Undo", "✋=Freeze", "☝=Draw",
        ]
        x = 8
        for h in hints:
            cv2.putText(frame, h, (x, WINDOW_H - 12),
                        FONT, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)
            x += len(h) * 8 + 12
            if x > WINDOW_W - 80:
                break

        depth = state.get("depth_cm", 0)
        cv2.putText(frame, f"Z:{depth:.0f}cm", (WINDOW_W - 80, WINDOW_H - 12),
                    FONT, 0.5, COLOR_ACC, 1, cv2.LINE_AA)
        return frame

    def _draw_gesture_popup(self, frame):
        elapsed = time.time() - self._gesture_time
        if not self._gesture_text or elapsed > self._gesture_dur:
            return frame
        alpha = max(0.0, 1.0 - elapsed / self._gesture_dur)
        text  = self._gesture_text
        tw, th = cv2.getTextSize(text, FONT_BOLD, 1.1, 2)[0]
        cx = (WINDOW_W - tw) // 2
        cy = WINDOW_H // 2 - 40

        overlay = frame.copy()
        cv2.rectangle(overlay, (cx - 14, cy - th - 10),
                      (cx + tw + 14, cy + 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0)
        col = tuple(int(c * alpha) for c in COLOR_MAIN)
        cv2.putText(frame, text, (cx, cy), FONT_BOLD, 1.1, col, 2, cv2.LINE_AA)
        return frame

    def _draw_crosshair(self, frame, state):
        tip = state.get("fingertip_screen")
        if tip is None:
            return frame
        x, y = tip
        r = 14
        cv2.line(frame, (x - r, y), (x + r, y), COLOR_ACC, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y - r), (x, y + r), COLOR_ACC, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, COLOR_MAIN, -1, cv2.LINE_AA)
        return frame

    def _draw_trail(self, frame, state):
        trail = state.get("finger_trail", [])
        if len(trail) < 2:
            return frame
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            color = (
                int(0   * alpha),
                int(220 * alpha),
                int(255 * alpha),
            )
            cv2.line(frame, trail[i-1], trail[i], color, 2, cv2.LINE_AA)
        return frame

    def _draw_depth_indicator(self, frame, state):
        depth = state.get("depth_cm", 40)
        bar_h = int(np.interp(depth, [10, 120], [WINDOW_H - 60, 50]))
        cv2.rectangle(frame, (WINDOW_W - 18, 50),
                      (WINDOW_W - 8, WINDOW_H - 50), (40, 40, 40), -1)
        cv2.rectangle(frame, (WINDOW_W - 18, bar_h),
                      (WINDOW_W - 8, WINDOW_H - 50), COLOR_MAIN, -1)
        return frame
