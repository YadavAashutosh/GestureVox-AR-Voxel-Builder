<div align="center">

# 🖐️ GestureVox — AI Hand Gesture AR Voxel Builder

### Build 3D worlds in Augmented Reality using only your hands — no mouse, no keyboard, no controller.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.33-green?style=for-the-badge)
![OpenGL](https://img.shields.io/badge/OpenGL-PyOpenGL-red?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Early%20Alpha-orange?style=for-the-badge)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge)

<br/>

> **⚠️ HONEST STATUS: The app runs and gestures work — but gesture accuracy and AR tracking still need a lot of work. This is an early alpha. Contributions, fixes, and improvements are very welcome!**

</div>

---

## 🎥 What Is This?

GestureVox is a real-time **Augmented Reality voxel builder** where:

- Your **webcam feed is the fullscreen background** — you see your real room
- **3D voxel blocks float on top** of your webcam feed using OpenGL
- **Your hands control everything** — place blocks, delete, undo, draw, zoom, rotate
- No mouse. No keyboard. No controller. Just your hands.

Built with **MediaPipe** (hand tracking), **OpenCV** (webcam + 2D overlays), **PyOpenGL + GLFW** (3D rendering), and **NumPy/SciPy** (gesture math).

---

## ✅ What Works

- Webcam feed as AR background
- 3D voxel rendering overlaid on webcam (OpenGL)
- Hand skeleton drawn on webcam feed (colored per finger)
- Basic gesture detection (pinch, fist, open palm, scroll, swipe)
- Place and delete voxels
- Undo / Redo (100 levels)
- 22 voxel types with distinct colors
- Flood fill
- Air drawing mode (freehand)
- Color picker wheel overlay
- Shape library panel (21 procedural shapes)
- JSON world save / load
- Procedural starter floor on launch
- Chunk-based spatial indexing
- Box select / copy / paste

---

## ⚠️ Known Bugs & Limitations

> This project is in **early alpha**. Here is what is broken or incomplete:

| Issue | Details |
|---|---|
| **Gesture accuracy is low** | Gestures misfire often, especially swipes and rotates |
| **Laggy on most machines** | MediaPipe is CPU-heavy; no GPU acceleration yet |
| **AR depth is fake** | Voxels don't truly align to real-world surfaces |
| **No floor plane detection** | Voxels float on a fixed grid, not on real surfaces |
| **Camera calibration is estimated** | Not using real webcam intrinsics |
| **Gesture cooldowns feel off** | Some gestures trigger too fast or too slow |
| **Double pinch is unreliable** | Timing window is hard to hit consistently |
| **Draw mode is rough** | Air drawing path is jittery |
| **No collision detection** | Voxels can overlap |
| **Windows-only tested** | Linux/Mac not tested yet |
| **Python 3.14 warnings** | MediaPipe throws TFLite warnings (harmless but noisy) |

---

## 🙌 This Is Open Source — Please Contribute!

This project was built as a proof-of-concept. The architecture is solid but the controls need serious polish. **You can make this amazing.**

### 🔧 Good First Issues to Fix

- [ ] Improve gesture debouncing and confidence thresholds
- [ ] Add proper webcam intrinsic calibration
- [ ] Smooth out air drawing jitter (Kalman filter?)
- [ ] GPU-accelerate MediaPipe (use `model_asset_buffer` + GPU delegate)
- [ ] Add real floor plane detection (MediaPipe Objectron or depth estimation)
- [ ] Fix double-pinch timing
- [ ] Test and fix on Linux / macOS
- [ ] Add gesture training / customization UI
- [ ] Performance profiling and optimization
- [ ] Add multiplayer / network sync (ambitious!)

### How to Contribute

1. Fork this repo
2. Create a branch: `git checkout -b fix/gesture-accuracy`
3. Make your changes
4. Open a Pull Request with a clear description of what you fixed

All skill levels welcome. Even fixing one bug helps!

---

## 🚀 Quick Start

### Requirements

- Python 3.10, 3.11, 3.12, or 3.13 (3.14 works but has warnings)
- A webcam
- Windows (Linux/Mac untested)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/GestureVox.git
cd GestureVox
pip install mediapipe==0.10.33 opencv-python PyOpenGL glfw numpy scipy Pillow
python generate_ar_voxel_builder.py
cd ar_voxel_builder
python main.py
```

> On first run, the app will automatically download the MediaPipe hand landmark model (~4 MB).

---

## 🖐️ Gesture Controls

| Gesture | Action |
|---|---|
| Pinch (thumb + index) | Place voxel |
| Double pinch | Delete voxel |
| Fist | Freeze cursor |
| Open palm | Toggle shape panel |
| Index finger only | Air draw mode |
| 2-finger scroll up/down | Cycle voxel type |
| 3-finger swipe left | Undo |
| 3-finger swipe right | Redo |
| 4-finger tap | Flood fill |
| Pinky raise | Color picker |
| Wrist rotate left/right | Rotate camera |
| 2-hand spread apart | Zoom out |
| 2-hand pinch together | Zoom in |
| Both fists | Reset camera view |
| L-shape hand | Box select mode |
| OK gesture | Copy selection |
| Peace sign | Paste selection |
| Thumb up | Save world |
| Thumb down | Load last world |
| Rock sign | Toggle lighting |
| Hand tilt left/right | Pan camera |
| Slow pinch hold 2s | Eyedropper (pick color) |

---

## 📁 Project Structure

```
ar_voxel_builder/
├── main.py                        # Entry point, app orchestrator
├── gesture_engine/
│   ├── hand_tracker.py            # MediaPipe Hands wrapper (new Tasks API)
│   └── gesture_classifier.py     # 23-gesture geometric classifier
├── drawing_engine/
│   └── air_drawing.py             # Air drawing, Bezier curves, plane lock
├── shape_library/
│   ├── shapes.py                  # 21 procedural shape generators
│   └── stamper.py                 # Places shapes into voxel world
├── voxel_core/
│   ├── voxel.py                   # Voxel dataclass
│   └── voxel_world.py             # World state, undo/redo, flood fill
├── renderer/
│   ├── gl_renderer.py             # PyOpenGL GLSL shader pipeline
│   └── ar_compositor.py           # Blends OpenGL output onto webcam frame
├── ui/
│   ├── hud.py                     # OpenCV HUD overlays
│   ├── color_picker.py            # HSV color wheel
│   └── shape_panel.py             # Shape library panel
└── utils/
    ├── constants.py               # All tunable parameters
    ├── math_helpers.py            # 3D math, projection, depth estimation
    └── save_load.py               # JSON world persistence
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| MediaPipe 0.10.33 | Real-time 21-point hand landmark detection |
| OpenCV | Webcam capture, 2D overlays, display |
| PyOpenGL + GLFW | 3D voxel rendering with GLSL shaders |
| NumPy | Gesture math, 3D coordinate mapping |
| SciPy | Bezier curve fitting for air drawing |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Credits & Acknowledgements

- Built by **Aashutosh** as a personal AI + AR experiment
- Hand tracking powered by [MediaPipe](https://mediapipe.dev/) by Google
- 3D rendering via [PyOpenGL](https://pyopengl.sourceforge.net/)
- If you improve this project, please give a star and open a PR!

---

<div align="center">

**If this project helped you or made you smile, please give it a ⭐**

*"It works. It's rough. Make it better."*

</div>
