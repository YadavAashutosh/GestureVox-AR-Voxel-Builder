"""Color picker wheel overlay drawn with OpenCV."""
import cv2
import numpy as np
import math


class ColorPickerWheel:
    def __init__(self, cx: int, cy: int, radius: int = 90):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.visible = False
        self.selected_color: tuple[int,int,int] = (255, 255, 255)
        self._wheel_img = self._build_wheel()

    def _build_wheel(self) -> np.ndarray:
        r = self.radius
        img = np.zeros((r*2+1, r*2+1, 4), dtype=np.uint8)
        for y in range(r*2+1):
            for x in range(r*2+1):
                dx, dy = x - r, y - r
                dist = math.hypot(dx, dy)
                if dist <= r:
                    angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
                    sat   = dist / r
                    rgb   = self._hsv_to_rgb(angle, sat, 1.0)
                    img[y, x] = [rgb[2], rgb[1], rgb[0], 220]
        return img

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
        return int(r*255), int(g*255), int(b*255)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self.visible:
            return frame
        r = self.radius
        x0 = self.cx - r
        y0 = self.cy - r
        x1 = x0 + r*2 + 1
        y1 = y0 + r*2 + 1
        # Clamp to frame
        fx0 = max(x0, 0); fy0 = max(y0, 0)
        fx1 = min(x1, frame.shape[1]); fy1 = min(y1, frame.shape[0])
        wx0 = fx0 - x0; wy0 = fy0 - y0
        wx1 = wx0 + (fx1 - fx0); wy1 = wy0 + (fy1 - fy0)
        if fx1 <= fx0 or fy1 <= fy0:
            return frame
        wheel_crop = self._wheel_img[wy0:wy1, wx0:wx1]
        alpha = wheel_crop[:, :, 3:4].astype(np.float32) / 255
        frame[fy0:fy1, fx0:fx1] = (
            frame[fy0:fy1, fx0:fx1] * (1 - alpha) +
            wheel_crop[:, :, :3] * alpha
        ).astype(np.uint8)
        cv2.circle(frame, (self.cx, self.cy), r, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def pick(self, sx: int, sy: int) -> tuple[int,int,int] | None:
        if not self.visible:
            return None
        dx, dy = sx - self.cx, sy - self.cy
        dist = math.hypot(dx, dy)
        if dist > self.radius:
            return None
        angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
        sat   = dist / self.radius
        rgb   = self._hsv_to_rgb(angle, sat, 1.0)
        self.selected_color = rgb
        return rgb

    def toggle(self):
        self.visible = not self.visible
