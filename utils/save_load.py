"""JSON world persistence."""
import json, os, time
from pathlib import Path

SAVE_DIR = Path("saves")


def save_world(voxel_world, filename: str | None = None) -> str:
    SAVE_DIR.mkdir(exist_ok=True)
    if filename is None:
        filename = f"world_{int(time.time())}.json"
    path = SAVE_DIR / filename
    data = {
        "version": 1,
        "voxels": [
            {"x": x, "y": y, "z": z, "type": t, "color": list(c)}
            for (x, y, z), (t, c) in voxel_world.voxels.items()
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


def load_world(voxel_world, filename: str) -> bool:
    path = SAVE_DIR / filename
    if not path.exists():
        return False
    with open(path) as f:
        data = json.load(f)
    voxel_world.clear()
    for v in data.get("voxels", []):
        voxel_world.set_voxel(
            (v["x"], v["y"], v["z"]),
            v["type"],
            tuple(v["color"])
        )
    return True


def list_saves() -> list[str]:
    SAVE_DIR.mkdir(exist_ok=True)
    return sorted([p.name for p in SAVE_DIR.glob("*.json")])
