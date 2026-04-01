"""Composites the OpenGL framebuffer on top of the webcam frame."""
import numpy as np
import cv2
from OpenGL.GL import *


class ARCompositor:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._fbo = None
        self._tex = None
        self._rbo = None

    def init_fbo(self):
        """Create off-screen FBO for GL rendering."""
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        self._tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self._tex, 0)

        self._rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.w, self.h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, self._rbo)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.w, self.h)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def read_pixels(self) -> np.ndarray:
        """Read RGBA pixels from FBO as numpy array (H, W, 4)."""
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        buf = glReadPixels(0, 0, self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 4)
        return np.flipud(img)

    def composite(self, webcam_bgr: np.ndarray, gl_rgba: np.ndarray) -> np.ndarray:
        """Alpha-blend GL layer onto webcam frame."""
        gl_bgr  = cv2.cvtColor(gl_rgba, cv2.COLOR_RGBA2BGRA)
        alpha   = gl_bgr[:, :, 3:4].astype(np.float32) / 255.0
        gl_rgb  = gl_bgr[:, :, :3].astype(np.float32)
        cam_rgb = webcam_bgr.astype(np.float32)
        out = cam_rgb * (1 - alpha) + gl_rgb * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
