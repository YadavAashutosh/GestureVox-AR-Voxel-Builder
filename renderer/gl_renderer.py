"""OpenGL voxel renderer using PyOpenGL + GLFW."""
import numpy as np
import ctypes
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders

from utils.constants import (
    WINDOW_W, WINDOW_H, VOXEL_SIZE, FOV_Y,
    NEAR_PLANE, FAR_PLANE, GHOST_COLOR, VOXEL_COLORS_GL
)
from utils.math_helpers import (
    voxel_to_world, perspective_matrix,
    rot_matrix_y, rot_matrix_x, translation_matrix
)

# ── GLSL shaders ─────────────────────────────
VERT_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec4 aColor;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform vec3 uLightDir;
uniform bool uLighting;

out vec4 vColor;

void main(){
    gl_Position = uMVP * vec4(aPos, 1.0);
    if(uLighting){
        vec3 N = normalize(mat3(uModel) * aNormal);
        float diff = max(dot(N, normalize(uLightDir)), 0.0);
        float amb  = 0.35;
        vColor = vec4(aColor.rgb * (amb + diff * 0.65), aColor.a);
    } else {
        vColor = aColor;
    }
}
"""

FRAG_SRC = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main(){
    FragColor = vColor;
}
"""

FACE_NORMALS = [
    np.array([ 0,  0,  1], dtype=np.float32),  # front
    np.array([ 0,  0, -1], dtype=np.float32),  # back
    np.array([-1,  0,  0], dtype=np.float32),  # left
    np.array([ 1,  0,  0], dtype=np.float32),  # right
    np.array([ 0,  1,  0], dtype=np.float32),  # top
    np.array([ 0, -1,  0], dtype=np.float32),  # bottom
]

# 6 faces × 2 triangles × 3 vertices
CUBE_FACE_VERTS = [
    # front (z+)
    [(-1,-1, 1),( 1,-1, 1),( 1, 1, 1),(-1,-1, 1),( 1, 1, 1),(-1, 1, 1)],
    # back  (z-)
    [( 1,-1,-1),(-1,-1,-1),(-1, 1,-1),( 1,-1,-1),(-1, 1,-1),( 1, 1,-1)],
    # left  (x-)
    [(-1,-1,-1),(-1,-1, 1),(-1, 1, 1),(-1,-1,-1),(-1, 1, 1),(-1, 1,-1)],
    # right (x+)
    [( 1,-1, 1),( 1,-1,-1),( 1, 1,-1),( 1,-1, 1),( 1, 1,-1),( 1, 1, 1)],
    # top   (y+)
    [(-1, 1, 1),( 1, 1, 1),( 1, 1,-1),(-1, 1, 1),( 1, 1,-1),(-1, 1,-1)],
    # bottom(y-)
    [(-1,-1,-1),( 1,-1,-1),( 1,-1, 1),(-1,-1,-1),( 1,-1, 1),(-1,-1, 1)],
]


class GLRenderer:
    def __init__(self, voxel_world):
        self.world   = voxel_world
        self.shader  = None
        self.vao = self.vbo = 0

        # Camera
        self.cam_yaw   = 0.0
        self.cam_pitch = 0.0
        self.cam_zoom  = 1.0
        self.cam_pan_x = 0.0
        self.cam_pan_y = 0.0
        self.lighting  = True

        self._vertex_data: np.ndarray | None = None
        self._n_verts = 0

    def init_gl(self):
        self.shader = gl_shaders.compileProgram(
            gl_shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER),
            gl_shaders.compileShader(FRAG_SRC, GL_FRAGMENT_SHADER),
        )
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        stride = (3 + 3 + 4) * 4  # pos + normal + color (floats)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _build_mesh(self, voxels: dict, extra_voxels: list | None = None,
                    ghost_coords: list | None = None) -> np.ndarray:
        """Build interleaved vertex buffer: pos(3) + normal(3) + color(4)."""
        rows = []
        hs = VOXEL_SIZE / 2

        def add_voxel(coord, color4):
            wx, wy, wz = voxel_to_world(*coord)
            for face_verts, normal in zip(CUBE_FACE_VERTS, FACE_NORMALS):
                for v in face_verts:
                    px = wx + v[0] * hs
                    py = wy + v[1] * hs
                    pz = wz + v[2] * hs
                    rows.append([px, py, pz,
                                 normal[0], normal[1], normal[2],
                                 color4[0], color4[1], color4[2], color4[3]])

        for coord, (vtype, _) in voxels.items():
            if vtype == 0:
                continue
            c = VOXEL_COLORS_GL.get(vtype, (0.7, 0.7, 0.7, 1.0))
            add_voxel(coord, c)

        if ghost_coords:
            gc = GHOST_COLOR
            for coord in ghost_coords:
                add_voxel(coord, gc)

        if not rows:
            return np.zeros((0, 10), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def upload_mesh(self, ghost_coords: list | None = None):
        data = self._build_mesh(self.world.voxels, ghost_coords=ghost_coords)
        self._vertex_data = data
        self._n_verts = len(data)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        if self._n_verts > 0:
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        self.world._dirty = False

    def get_mvp(self) -> np.ndarray:
        proj = perspective_matrix(FOV_Y, WINDOW_W / WINDOW_H, NEAR_PLANE, FAR_PLANE)
        view = (translation_matrix(self.cam_pan_x, self.cam_pan_y, -80 * self.cam_zoom)
                @ rot_matrix_x(self.cam_pitch)
                @ rot_matrix_y(self.cam_yaw))
        return proj @ view

    def render(self, ghost_coords: list | None = None):
        if self.world._dirty or ghost_coords is not None:
            self.upload_mesh(ghost_coords)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0, 0, 0, 0)

        if self._n_verts == 0:
            return

        glUseProgram(self.shader)
        mvp  = self.get_mvp()
        model = np.eye(4, dtype=np.float32)
        light = np.array([0.5, 1.0, 0.7], dtype=np.float32)

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "uMVP"),
                           1, GL_TRUE, mvp)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "uModel"),
                           1, GL_TRUE, model)
        glUniform3fv(glGetUniformLocation(self.shader, "uLightDir"),
                     1, light)
        glUniform1i(glGetUniformLocation(self.shader, "uLighting"),
                    int(self.lighting))

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self._n_verts)
        glBindVertexArray(0)

    def raycast(self, ray_origin: np.ndarray,
                ray_dir: np.ndarray, max_dist=200) -> tuple | None:
        """DDA voxel raycast. Returns (coord, hit_world_pos) or None."""
        from utils.math_helpers import world_to_voxel, voxel_to_world
        from utils.constants import VOXEL_SIZE, GRID_ORIGIN
        pos = ray_origin.copy().astype(np.float64)
        step = ray_dir / np.linalg.norm(ray_dir) * (VOXEL_SIZE * 0.25)
        for _ in range(int(max_dist / (VOXEL_SIZE * 0.25))):
            pos += step
            coord = world_to_voxel(pos.astype(np.float32))
            v = self.world.get_voxel(coord)
            if v and v[0] != 0:
                return coord, pos.copy()
        return None
