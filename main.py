#!/usr/bin/env python3
"""
AI-Powered Hand Gesture AR Voxel Builder
Entry point — launches webcam thread + render loop.
"""
import sys, os, time, threading, collections
import numpy as np
import cv2
import glfw
from OpenGL.GL import *

from utils.constants import (
    WINDOW_W, WINDOW_H, TARGET_FPS_WEBCAM, TARGET_FPS_RENDER,
    VOXEL_TYPES, GRID_ORIGIN, VOXEL_SIZE
)
from utils.math_helpers import (
    get_camera_matrix, estimate_depth_from_palm,
    world_to_voxel, voxel_to_world, lerp
)
from utils.save_load import save_world, load_world, list_saves

from voxel_core.voxel_world import VoxelWorld
from gesture_engine.hand_tracker import HandTracker
from gesture_engine.gesture_classifier import GestureClassifier, Gesture
from drawing_engine.air_drawing import AirDrawEngine
from shape_library.stamper import ShapeStamper
from renderer.gl_renderer import GLRenderer
from renderer.ar_compositor import ARCompositor
from ui.hud import HUDOverlay
from ui.color_picker import ColorPickerWheel
from ui.shape_panel import ShapePanel


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.webcam_frame = None
        self.current_gesture = Gesture.NONE
        self.fingertip_screen = None
        self.fingertip_world  = None
        self.depth_cm = 40.0
        self.finger_trail = collections.deque(maxlen=20)
        self.voxel_type = 1
        self.current_color = VOXEL_TYPES[1][1]
        self.ghost_coords = []
        self.box_select_start = None
        self.clipboard = {}
        self.mode = "PLACE"
        self.lighting = True
        self.running  = True
        self.n_voxels = 0
        self.cam_yaw   = 0.0
        self.cam_pitch = 0.0
        self.cam_zoom  = 1.0
        self.cam_pan_x = 0.0
        self.cam_pan_y = 0.0
        self._last_action = {}

    def cooldown_ok(self, g, secs=0.5):
        now = time.time()
        if now - self._last_action.get(g, 0.0) >= secs:
            self._last_action[g] = now
            return True
        return False


def build_starter_floor(world):
    for x in range(-6, 7):
        for z in range(-6, 7):
            vt = 2 if (x + z) % 3 != 0 else 1
            world.set_voxel((x, -1, z), vt)


def webcam_thread(state, world, draw_engine, stamper,
                  hud, color_picker, shape_panel):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker    = HandTracker(max_hands=2)
    classifier = GestureClassifier()
    cam_mat    = get_camera_matrix()

    frame_dt = 1.0 / 30
    prev_t   = time.time()
    # Process MediaPipe every N frames for speed
    mp_every = 2
    frame_count = 0

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        now   = time.time()
        dt    = now - prev_t
        prev_t = now
        frame_count += 1

        all_hands = []
        lms = None

        # Only run MediaPipe every 2nd frame
        if frame_count % mp_every == 0:
            small = cv2.resize(frame, (640, 360))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            tracker.process(rgb)

        all_hands = tracker.get_all_hands()

        gesture = Gesture.NONE

        if all_hands:
            lms, label = all_hands[0]
            gesture = classifier.classify(all_hands, dt)

            depth  = estimate_depth_from_palm(lms)
            tip_sx = int(lms[8].x * WINDOW_W)
            tip_sy = int(lms[8].y * WINDOW_H)
            tip_world = classifier.get_fingertip_3d(lms, depth)

            with state.lock:
                state.depth_cm         = depth
                state.fingertip_screen = (tip_sx, tip_sy)
                state.fingertip_world  = tip_world
                state.current_gesture  = gesture
                state.finger_trail.append((tip_sx, tip_sy))

            frame = tracker.draw_skeleton(frame, 0)

            if state.mode == "DRAW":
                draw_engine.update(tip_world, speed=1.0)
            else:
                if draw_engine.active:
                    draw_engine.stop()

            gx, gy, gz = world_to_voxel(tip_world)
            ghost = [(gx, gy, gz)]
            with state.lock:
                state.ghost_coords = ghost
        else:
            with state.lock:
                state.fingertip_screen = None
                state.current_gesture  = Gesture.NONE
                state.finger_trail.clear()
                state.ghost_coords     = []

        _handle_gesture(gesture, state, world, draw_engine,
                        stamper, hud, color_picker, shape_panel, lms)

        with state.lock:
            state.n_voxels = len(world.voxels)
            vt       = state.voxel_type
            mode     = state.mode
            depth_cm = state.depth_cm
            ftip     = state.fingertip_screen
            trail    = list(state.finger_trail)
            lighting = state.lighting

        hud_state = {
            "voxel_type": vt, "mode": mode,
            "n_voxels": state.n_voxels, "lighting": lighting,
            "depth_cm": depth_cm, "fingertip_screen": ftip,
            "finger_trail": trail,
        }
        frame = hud.draw(frame, hud_state)
        frame = color_picker.draw(frame)
        frame = shape_panel.draw(frame)

        with state.lock:
            state.webcam_frame = frame.copy()

        elapsed = time.time() - now
        sleep_t = frame_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    cap.release()
    tracker.close()


def _handle_gesture(gesture, state, world, draw_engine,
                    stamper, hud, color_picker, shape_panel, lms):
    if gesture == Gesture.NONE or lms is None:
        return

    with state.lock:
        tip_w = state.fingertip_world
        vt    = state.voxel_type
        color = state.current_color
        mode  = state.mode

    # Safely check tip_w is not None and is valid array
    def has_tip():
        return tip_w is not None and isinstance(tip_w, np.ndarray)

    def flash(txt):
        hud.set_gesture(txt)

    if gesture == Gesture.PINCH and state.cooldown_ok(gesture, 0.35):
        if mode == "PLACE" and has_tip():
            coord = world_to_voxel(tip_w)
            world.set_voxel(coord, vt, color)
            flash(f"PLACED: {VOXEL_TYPES[vt][0]}")
        elif mode == "DRAW":
            draw_engine.confirm()
            flash("DRAW CONFIRMED")

    elif gesture == Gesture.DOUBLE_PINCH and state.cooldown_ok(gesture, 0.4):
        if has_tip():
            coord = world_to_voxel(tip_w)
            world.remove_voxel(coord)
            flash("DELETED")

    elif gesture == Gesture.FIST and state.cooldown_ok(gesture, 0.5):
        with state.lock:
            state.fingertip_screen = None
        flash("CURSOR FROZEN")

    elif gesture == Gesture.OPEN_PALM and state.cooldown_ok(gesture, 0.6):
        shape_panel.toggle()
        flash("SHAPE PANEL " + ("ON" if shape_panel.visible else "OFF"))

    elif gesture == Gesture.DRAW_MODE:
        if mode != "DRAW":
            with state.lock:
                state.mode = "DRAW"
            draw_engine.start()
            flash("DRAW MODE")

    elif gesture == Gesture.TWO_FINGER_SCROLL_UP and state.cooldown_ok(gesture, 0.25):
        with state.lock:
            state.voxel_type = (state.voxel_type % (len(VOXEL_TYPES) - 1)) + 1
        flash(f"VOXEL: {VOXEL_TYPES[state.voxel_type][0]}")

    elif gesture == Gesture.TWO_FINGER_SCROLL_DOWN and state.cooldown_ok(gesture, 0.25):
        with state.lock:
            state.voxel_type = max(1, state.voxel_type - 1)
        flash(f"VOXEL: {VOXEL_TYPES[state.voxel_type][0]}")

    elif gesture == Gesture.THREE_SWIPE_LEFT and state.cooldown_ok(gesture, 0.4):
        world.undo(); flash("UNDO")

    elif gesture == Gesture.THREE_SWIPE_RIGHT and state.cooldown_ok(gesture, 0.4):
        world.redo(); flash("REDO")

    elif gesture == Gesture.FOUR_FINGER_TAP and state.cooldown_ok(gesture, 0.6):
        if has_tip():
            world.flood_fill(world_to_voxel(tip_w), vt, color)
            flash("FLOOD FILL")

    elif gesture == Gesture.PINKY_RAISE and state.cooldown_ok(gesture, 0.5):
        color_picker.toggle()
        flash("COLOR PICKER " + ("ON" if color_picker.visible else "OFF"))

    elif gesture == Gesture.WRIST_ROTATE_LEFT and state.cooldown_ok(gesture, 0.15):
        with state.lock: state.cam_yaw -= 10.0
        flash("ROTATE LEFT")

    elif gesture == Gesture.WRIST_ROTATE_RIGHT and state.cooldown_ok(gesture, 0.15):
        with state.lock: state.cam_yaw += 10.0
        flash("ROTATE RIGHT")

    elif gesture == Gesture.TWO_HAND_SPREAD and state.cooldown_ok(gesture, 0.2):
        with state.lock: state.cam_zoom = min(state.cam_zoom + 0.1, 4.0)
        flash("ZOOM OUT")

    elif gesture == Gesture.TWO_HAND_PINCH and state.cooldown_ok(gesture, 0.2):
        with state.lock: state.cam_zoom = max(state.cam_zoom - 0.1, 0.3)
        flash("ZOOM IN")

    elif gesture == Gesture.BOTH_FISTS and state.cooldown_ok(gesture, 0.8):
        with state.lock:
            state.cam_yaw = state.cam_pitch = state.cam_pan_x = state.cam_pan_y = 0.0
            state.cam_zoom = 1.0
        flash("VIEW RESET")

    elif gesture == Gesture.L_SHAPE and state.cooldown_ok(gesture, 0.5):
        with state.lock:
            if state.mode != "SELECT":
                state.mode = "SELECT"
                state.box_select_start = world_to_voxel(tip_w) if has_tip() else None
                flash("BOX SELECT START")
            else:
                state.mode = "PLACE"
                flash("BOX SELECT END")

    elif gesture == Gesture.OK_GESTURE and state.cooldown_ok(gesture, 0.5):
        with state.lock:
            if state.box_select_start is not None and has_tip():
                c2 = world_to_voxel(tip_w)
                state.clipboard = world.get_region(state.box_select_start, c2)
                flash(f"COPIED {len(state.clipboard)} voxels")

    elif gesture == Gesture.PEACE and state.cooldown_ok(gesture, 0.5):
        with state.lock:
            cb = state.clipboard.copy()
        if cb and has_tip():
            world.paste_region(cb, world_to_voxel(tip_w))
            flash(f"PASTED {len(cb)} voxels")

    elif gesture == Gesture.THUMB_UP and state.cooldown_ok(gesture, 1.5):
        path = save_world(world)
        flash(f"SAVED")

    elif gesture == Gesture.THUMB_DOWN and state.cooldown_ok(gesture, 1.5):
        saves = list_saves()
        if saves:
            load_world(world, saves[-1])
            flash(f"LOADED: {saves[-1]}")
        else:
            flash("NO SAVES FOUND")

    elif gesture == Gesture.ROCK_SIGN and state.cooldown_ok(gesture, 0.6):
        with state.lock: state.lighting = not state.lighting
        flash("LIGHTING " + ("ON" if state.lighting else "OFF"))

    elif gesture == Gesture.HAND_TILT_LEFT and state.cooldown_ok(gesture, 0.15):
        with state.lock: state.cam_pan_x -= 2.0
    elif gesture == Gesture.HAND_TILT_RIGHT and state.cooldown_ok(gesture, 0.15):
        with state.lock: state.cam_pan_x += 2.0

    elif gesture == Gesture.SLOW_PINCH_HOLD and state.cooldown_ok(gesture, 2.0):
        if has_tip():
            v = world.get_voxel(world_to_voxel(tip_w))
            if v:
                with state.lock:
                    state.voxel_type   = v[0]
                    state.current_color = v[1]
                flash(f"EYEDROPPER: {VOXEL_TYPES[v[0]][0]}")


def main():
    world       = VoxelWorld()
    build_starter_floor(world)
    state       = AppState()
    draw_engine = AirDrawEngine(world,
                                lambda: state.voxel_type,
                                lambda: state.current_color)
    stamper     = ShapeStamper(world)
    hud         = HUDOverlay()
    color_picker = ColorPickerWheel(cx=WINDOW_W - 110, cy=WINDOW_H // 2)
    shape_panel  = ShapePanel()

    if not glfw.init():
        print("GLFW init failed"); sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DECORATED, False)

    window = glfw.create_window(WINDOW_W, WINDOW_H, "AR Voxel Builder", None, None)
    if not window:
        glfw.terminate(); sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(0)   # vsync off for speed

    renderer   = GLRenderer(world)
    compositor = ARCompositor(WINDOW_W, WINDOW_H)
    renderer.init_gl()
    compositor.init_fbo()

    wt = threading.Thread(
        target=webcam_thread,
        args=(state, world, draw_engine, stamper,
              hud, color_picker, shape_panel),
        daemon=True,
    )
    wt.start()

    print("=" * 45)
    print("  AR Voxel Builder — RUNNING")
    print("  Press ESC to quit")
    print("=" * 45)

    frame_dt = 1.0 / TARGET_FPS_RENDER

    while not glfw.window_should_close(window) and state.running:
        t0 = time.time()

        with state.lock:
            webcam_bgr = state.webcam_frame
            ghost      = state.ghost_coords[:]
            cam_yaw    = state.cam_yaw
            cam_pitch  = state.cam_pitch
            cam_zoom   = state.cam_zoom
            cam_pan_x  = state.cam_pan_x
            cam_pan_y  = state.cam_pan_y
            lighting   = state.lighting
            draw_prev  = list(draw_engine.preview_voxels)

        renderer.cam_yaw   = cam_yaw
        renderer.cam_pitch = cam_pitch
        renderer.cam_zoom  = cam_zoom
        renderer.cam_pan_x = cam_pan_x
        renderer.cam_pan_y = cam_pan_y
        renderer.lighting  = lighting

        all_ghosts = list(set(map(tuple, ghost)) | set(map(tuple, draw_prev)))

        compositor.bind()
        renderer.render(ghost_coords=all_ghosts if all_ghosts else None)
        compositor.unbind()
        gl_pixels = compositor.read_pixels()

        if webcam_bgr is not None and gl_pixels is not None:
            composed = compositor.composite(webcam_bgr, gl_pixels)
        elif webcam_bgr is not None:
            composed = webcam_bgr
        else:
            composed = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

        cv2.imshow("AR Voxel Builder", composed)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            state.running = False
            break

        glfw.poll_events()

        elapsed = time.time() - t0
        sleep_t = frame_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    state.running = False
    wt.join(timeout=2)
    glfw.terminate()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()