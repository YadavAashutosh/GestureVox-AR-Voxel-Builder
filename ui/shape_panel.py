"""Shape library panel overlay."""
import cv2
import numpy as np
from shape_library.shapes import SHAPE_REGISTRY

FONT = cv2.FONT_HERSHEY_SIMPLEX


class ShapePanel:
    def __init__(self):
        self.visible = False
        self.selected: str | None = None
        self.names = list(SHAPE_REGISTRY.keys())
        self._cols = 3
        self._cell_w = 130
        self._cell_h = 36
        self._ox = 20
        self._oy = 60

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self.visible:
            return frame
        overlay = frame.copy()
        panel_w = self._cols * self._cell_w + 20
        panel_h = (len(self.names) // self._cols + 1) * self._cell_h + 20
        cv2.rectangle(overlay, (self._ox, self._oy),
                      (self._ox + panel_w, self._oy + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "SHAPE LIBRARY", (self._ox + 6, self._oy + 20),
                    FONT, 0.55, (0, 255, 170), 1, cv2.LINE_AA)
        for i, name in enumerate(self.names):
            col = i % self._cols
            row = i // self._cols
            x = self._ox + 10 + col * self._cell_w
            y = self._oy + 30 + row * self._cell_h
            selected = (name == self.selected)
            color = (0, 255, 170) if selected else (180, 180, 180)
            if selected:
                cv2.rectangle(frame, (x-4, y-18), (x + self._cell_w - 8, y + 8),
                              (0, 80, 50), -1)
            cv2.putText(frame, name, (x, y), FONT, 0.42, color, 1, cv2.LINE_AA)
        return frame

    def hit_test(self, sx: int, sy: int) -> str | None:
        if not self.visible:
            return None
        for i, name in enumerate(self.names):
            col = i % self._cols
            row = i // self._cols
            x = self._ox + 10 + col * self._cell_w
            y = self._oy + 30 + row * self._cell_h
            if x-4 <= sx <= x + self._cell_w - 8 and y - 18 <= sy <= y + 8:
                self.selected = name
                return name
        return None

    def toggle(self):
        self.visible = not self.visible
