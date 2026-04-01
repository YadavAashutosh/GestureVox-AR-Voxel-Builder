"""Procedural shape generators returning list of (dx,dy,dz) offsets."""
import math
from typing import List, Tuple

Coord = Tuple[int, int, int]


def _sphere(r: int) -> List[Coord]:
    pts = []
    for x in range(-r, r+1):
        for y in range(-r, r+1):
            for z in range(-r, r+1):
                if x*x + y*y + z*z <= r*r:
                    pts.append((x, y, z))
    return pts


def shape_cube(size=3) -> List[Coord]:
    h = size // 2
    return [(x, y, z) for x in range(-h, h+1)
                       for y in range(0, size)
                       for z in range(-h, h+1)]


def shape_sphere(radius=3) -> List[Coord]:
    return _sphere(radius)


def shape_dome(radius=3) -> List[Coord]:
    return [(x, y, z) for x, y, z in _sphere(radius) if y >= 0]


def shape_cylinder(radius=2, height=4) -> List[Coord]:
    pts = []
    for y in range(height):
        for x in range(-radius, radius+1):
            for z in range(-radius, radius+1):
                if x*x + z*z <= radius*radius:
                    pts.append((x, y, z))
    return pts


def shape_pyramid(base=5) -> List[Coord]:
    pts = []
    for y in range(base):
        r = base - y - 1
        for x in range(-r, r+1):
            for z in range(-r, r+1):
                pts.append((x, y, z))
    return pts


def shape_wall(width=6, height=4, depth=1) -> List[Coord]:
    hw = width // 2
    return [(x, y, z) for x in range(-hw, hw+1)
                       for y in range(height)
                       for z in range(depth)]


def shape_staircase(steps=5) -> List[Coord]:
    pts = []
    for s in range(steps):
        for x in range(3):
            for z in range(s + 1):
                pts.append((x - 1, z, s))
    return pts


def shape_arch(span=6, height=4) -> List[Coord]:
    pts = []
    hw = span // 2
    for x in range(-hw, hw+1):
        for y in range(height + 1):
            if abs(x) == hw or (y == height and abs(x) <= hw):
                pts.append((x, y, 0))
            # curved top
            r = hw
            if x*x + (y - height)*(y - height) <= r*r and y >= height - hw:
                pts.append((x, y, 0))
    return list(set(pts))


def shape_torus(R=4, r=2) -> List[Coord]:
    pts = []
    for x in range(-R-r, R+r+1):
        for y in range(-r, r+1):
            for z in range(-R-r, R+r+1):
                dist_ring = math.sqrt(x*x + z*z) - R
                if dist_ring*dist_ring + y*y <= r*r:
                    pts.append((x, y, z))
    return pts


def shape_tree(trunk_h=3, crown_r=2) -> List[Coord]:
    pts = [(0, y, 0) for y in range(trunk_h)]
    base_y = trunk_h
    pts += [(x, base_y + dy, z) for x, dy, z in _sphere(crown_r) if dy >= -1]
    return pts


def shape_castle_tower(radius=3, height=8) -> List[Coord]:
    pts = []
    for y in range(height):
        for x in range(-radius, radius+1):
            for z in range(-radius, radius+1):
                d = x*x + z*z
                inner = (radius-1)**2
                outer = radius**2
                if inner < d <= outer:
                    pts.append((x, y, z))
    # Battlements
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        bx = round(radius * math.cos(rad))
        bz = round(radius * math.sin(rad))
        pts.append((bx, height, bz))
    return list(set(pts))


def shape_bridge(length=8, width=3) -> List[Coord]:
    pts = []
    hw = width // 2
    # deck
    for z in range(length):
        for x in range(-hw, hw+1):
            pts.append((x, 0, z))
    # pillars
    for z in [0, length//2, length-1]:
        for y in range(-3, 0):
            for x in range(-hw, hw+1):
                pts.append((x, y, z))
    return list(set(pts))


def shape_checkerboard(size=6) -> List[Coord]:
    h = size // 2
    return [(x, 0, z) for x in range(-h, h+1)
                       for z in range(-h, h+1)
                       if (x + z) % 2 == 0]


def shape_spiral_staircase(steps=12, radius=3) -> List[Coord]:
    pts = []
    for s in range(steps):
        angle = math.radians(s * 30)
        x = round(radius * math.cos(angle))
        z = round(radius * math.sin(angle))
        pts.append((x, s, z))
        pts.append((0, s, 0))  # central pillar
    return pts


def shape_diamond(size=3) -> List[Coord]:
    pts = []
    for y in range(-size, size+1):
        r = size - abs(y)
        for x in range(-r, r+1):
            for z in range(-r, r+1):
                if abs(x) + abs(z) <= r:
                    pts.append((x, y + size, z))
    return pts


def shape_hollow_cube(size=4) -> List[Coord]:
    pts = []
    h = size // 2
    for x in range(-h, h+1):
        for y in range(0, size):
            for z in range(-h, h+1):
                on_face = (abs(x) == h or abs(z) == h or y == 0 or y == size-1)
                if on_face:
                    pts.append((x, y, z))
    return pts


def shape_ring(radius=4) -> List[Coord]:
    pts = []
    for x in range(-radius-1, radius+2):
        for z in range(-radius-1, radius+2):
            d = x*x + z*z
            if (radius-1)**2 <= d <= (radius+1)**2:
                pts.append((x, 0, z))
    return pts


def shape_cross(arm=4, height=5) -> List[Coord]:
    pts = []
    for y in range(height):
        for x in range(-arm, arm+1):
            pts.append((x, y, 0))
        for z in range(-arm, arm+1):
            pts.append((0, y, z))
    return list(set(pts))


def shape_ramp(length=6, width=3) -> List[Coord]:
    hw = width // 2
    pts = []
    for z in range(length):
        y = z
        for x in range(-hw, hw+1):
            pts.append((x, y, z))
    return pts


def shape_mushroom() -> List[Coord]:
    pts = [(0, y, 0) for y in range(4)]  # stem
    pts += shape_dome(radius=3)
    pts = [(x, y+4, z) for x, y, z in pts[4:]] + pts[:4]
    return pts


# Shape registry
SHAPE_REGISTRY = {
    "Cube":             shape_cube,
    "Sphere":           shape_sphere,
    "Dome":             shape_dome,
    "Cylinder":         shape_cylinder,
    "Pyramid":          shape_pyramid,
    "Wall":             shape_wall,
    "Staircase":        shape_staircase,
    "Arch":             shape_arch,
    "Torus":            shape_torus,
    "Tree":             shape_tree,
    "Castle Tower":     shape_castle_tower,
    "Bridge":           shape_bridge,
    "Checkerboard":     shape_checkerboard,
    "Spiral Staircase": shape_spiral_staircase,
    "Diamond":          shape_diamond,
    "Hollow Cube":      shape_hollow_cube,
    "Ring":             shape_ring,
    "Cross":            shape_cross,
    "Ramp":             shape_ramp,
    "Mushroom":         shape_mushroom,
    "Tower":            lambda: shape_cylinder(radius=2, height=10),
}
