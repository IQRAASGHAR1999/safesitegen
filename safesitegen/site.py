"""Site substrate: the geometric backbone a scenario is generated into.

In the full framework this object is populated from an IFC model or from a
photogrammetric / LiDAR capture of a real site. Here it is loaded from a small
template file with an identical schema, so the generator, the solver and the
validator are unchanged when a real capture is substituted.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from importlib import resources
from typing import Dict, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]


def _load_templates() -> Dict[str, dict]:
    with resources.files("safesitegen.knowledge").joinpath("site_templates.json").open() as fh:
        data = json.load(fh)
    return {t["id"]: t for t in data["templates"]}


TEMPLATES = _load_templates()


@dataclass
class Zone:
    name: str
    type: str
    x: int
    y: int
    w: int
    d: int

    def cells(self) -> List[Cell]:
        return [(x, y) for x in range(self.x, self.x + self.w) for y in range(self.y, self.y + self.d)]

    def contains(self, cx: float, cy: float) -> bool:
        return self.x <= cx < self.x + self.w and self.y <= cy < self.y + self.d


class SiteModel:
    """Coarse 1 m occupancy grid with zones, obstacles and an inspection route."""

    def __init__(self, spec: dict):
        self.id: str = spec["id"]
        self.name: str = spec["name"]
        self.provenance: str = spec.get("provenance", "unknown")
        self.width: int = spec["width"]
        self.depth: int = spec["depth"]
        self.grade_elevation_ft: float = spec.get("grade_elevation_ft", 0.0)
        self.deck_elevation_ft: float = spec.get("deck_elevation_ft", 0.0)
        self.zones: List[Zone] = [Zone(**z) for z in spec["zones"]]
        self.obstacles: List[dict] = spec.get("obstacles", [])
        self.entry: Cell = tuple(spec["entry"])  # type: ignore[assignment]
        self.patrol: List[Cell] = [tuple(p) for p in spec["patrol"]]  # type: ignore[misc]
        self._blocked = {(o["x"] + dx, o["y"] + dy)
                         for o in self.obstacles
                         for dx in range(o["w"])
                         for dy in range(o["d"])}
        # Static geometry, so every derived query is memoised once per site.
        self._reachable_cache: Dict[Cell, set] = {}
        self._visibility_cache: Dict[Cell, float] = {}
        self._patrol_distance_cache: Dict[Cell, float] = {}
        self._zone_cache: Dict[Cell, Optional[Zone]] = {}
        self._free_cells_cache: Dict[str, List[Cell]] = {}
        self._all_free_cache: Optional[List[Cell]] = None

    # ------------------------------------------------------------------ query
    _instances: Dict[str, "SiteModel"] = {}

    @classmethod
    def load(cls, template_id: str) -> "SiteModel":
        """Return a shared, read-only site instance.

        Site geometry never changes during generation, so instances are reused
        and their derived caches (reachability, sightlines, zone lookup) are paid
        for once per template rather than once per scenario.
        """
        if template_id not in TEMPLATES:
            raise KeyError(f"unknown site template {template_id!r}")
        if template_id not in cls._instances:
            cls._instances[template_id] = cls(TEMPLATES[template_id])
        return cls._instances[template_id]

    @classmethod
    def ids(cls) -> List[str]:
        return sorted(TEMPLATES)

    def in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x < self.width and 0 <= y < self.depth

    def is_blocked(self, x: int, y: int) -> bool:
        return (x, y) in self._blocked

    def walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not self.is_blocked(x, y)

    def zone_at(self, x: float, y: float) -> Optional[Zone]:
        key = (int(x), int(y))
        if key in self._zone_cache:
            return self._zone_cache[key]
        found = None
        for z in self.zones:
            if z.contains(x, y):
                found = z
                break
        self._zone_cache[key] = found
        return found

    def zones_of_type(self, zone_type: str) -> List[Zone]:
        return [z for z in self.zones if z.type == zone_type]

    def free_cells_of_type(self, zone_type: str) -> List[Cell]:
        if zone_type not in self._free_cells_cache:
            out: List[Cell] = []
            for z in self.zones_of_type(zone_type):
                out.extend(c for c in z.cells() if self.walkable(*c))
            self._free_cells_cache[zone_type] = out
        return self._free_cells_cache[zone_type]

    def all_free_cells(self) -> List[Cell]:
        if self._all_free_cache is None:
            self._all_free_cache = [(x, y) for x in range(self.width)
                                    for y in range(self.depth) if self.walkable(x, y)]
        return self._all_free_cache

    # ------------------------------------------------------------ reachability
    def reachable_cells(self, start: Optional[Cell] = None) -> set:
        start = start or self.entry
        if start in self._reachable_cache:
            return self._reachable_cache[start]
        if not self.walkable(*start):
            return set()
        seen = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) not in seen and self.walkable(nx, ny):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        self._reachable_cache[start] = seen
        return seen

    # --------------------------------------------------------------- sightline
    def has_line_of_sight(self, a: Cell, b: Cell) -> bool:
        """Bresenham traversal; obstacles occlude, endpoints do not."""
        x0, y0 = int(a[0]), int(a[1])
        x1, y1 = int(b[0]), int(b[1])
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while (x, y) != (x1, y1):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            if (x, y) == (x1, y1):
                break
            if self.is_blocked(x, y):
                return False
        return True

    def visibility_fraction(self, target: Cell, max_distance: float = 22.0) -> float:
        """Share of patrol viewpoints from which ``target`` is visible and near enough."""
        if not self.patrol:
            return 0.0
        key = (int(target[0]), int(target[1]))
        if key in self._visibility_cache:
            return self._visibility_cache[key]
        hits = 0
        for vp in self.patrol:
            d = ((vp[0] - target[0]) ** 2 + (vp[1] - target[1]) ** 2) ** 0.5
            if d <= max_distance and self.has_line_of_sight(vp, target):
                hits += 1
        frac = hits / len(self.patrol)
        self._visibility_cache[key] = frac
        return frac

    def nearest_patrol_distance(self, target: Cell) -> float:
        if not self.patrol:
            return float("inf")
        key = (int(target[0]), int(target[1]))
        if key not in self._patrol_distance_cache:
            self._patrol_distance_cache[key] = min(
                ((vp[0] - target[0]) ** 2 + (vp[1] - target[1]) ** 2) ** 0.5 for vp in self.patrol)
        return self._patrol_distance_cache[key]

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "provenance": self.provenance,
            "width": self.width,
            "depth": self.depth,
            "grade_elevation_ft": self.grade_elevation_ft,
            "deck_elevation_ft": self.deck_elevation_ft,
            "zones": [{"name": z.name, "type": z.type, "x": z.x, "y": z.y, "w": z.w, "d": z.d}
                      for z in self.zones],
            "obstacles": self.obstacles,
            "entry": list(self.entry),
            "patrol": [list(p) for p in self.patrol],
        }
