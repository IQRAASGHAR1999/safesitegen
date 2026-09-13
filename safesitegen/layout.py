"""Geometric realisation: place an abstract scenario graph into a real substrate.

Separating *what* the scenario teaches from *where* it sits is the reason the
system can be both generative and checkable. The semantic layer is produced by a
constrained generator; this module solves the placement as a constrained
optimisation over the site occupancy grid, in the spirit of layout solvers used
for language-guided 3D scene generation, but with regulatory and instructional
terms in the objective rather than only physical plausibility.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .knowledge import HAZARD_BY_ID
from .schema import Entity, Scenario
from .site import SiteModel

Cell = Tuple[int, int]

W_ZONE = 12.0
W_OVERLAP = 8.0
W_BLOCKED = 10.0
W_REACH = 6.0
W_VISIBILITY = 5.0
W_SEPARATION = 3.0
W_DIFFICULTY = 4.0

TARGET_VISIBILITY = 0.30
MIN_SEPARATION = 2.0


def _allowed_cells(entity: Entity, scenario: Scenario, site: SiteModel) -> List[Cell]:
    """Cells this entity may legally occupy."""
    hazard = next((h for h in scenario.hazards if h.target_entity == entity.id), None)
    if hazard is not None:
        meta = HAZARD_BY_ID.get(hazard.hazard_class)
        if meta:
            cells: List[Cell] = []
            for zt in meta["zone_types"]:
                cells.extend(site.free_cells_of_type(zt))
            if cells:
                return sorted(set(cells))
    return site.all_free_cells()


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def energy(scenario: Scenario, site: SiteModel, allowed: Dict[str, set]) -> float:
    """Total constraint energy. Lower is better; zero means fully satisfied."""
    e_total = 0.0
    occupied: Dict[Cell, int] = {}
    reachable = site.reachable_cells()

    for ent in scenario.entities:
        if not ent.placed:
            e_total += W_ZONE
            continue
        cell = (int(ent.x), int(ent.y))

        if not site.in_bounds(*cell):
            e_total += W_ZONE * 2
            continue
        if site.is_blocked(*cell):
            e_total += W_BLOCKED
        if cell not in allowed.get(ent.id, set()):
            e_total += W_ZONE
        if site.zone_at(*cell) is None:
            e_total += W_ZONE
        if ent.kind in ("worker", "equipment") and cell not in reachable:
            e_total += W_REACH
        occupied[cell] = occupied.get(cell, 0) + 1

    for count in occupied.values():
        if count > 1:
            e_total += W_OVERLAP * (count - 1)

    hazard_cells: List[Cell] = []
    for h in scenario.hazards:
        ent = scenario.entity(h.target_entity)
        if not ent.placed:
            continue
        cell = (int(ent.x), int(ent.y))
        hazard_cells.append(cell)
        if h.intended_detectable:
            vis = site.visibility_fraction(cell)
            if vis < TARGET_VISIBILITY:
                e_total += W_VISIBILITY * (TARGET_VISIBILITY - vis) / TARGET_VISIBILITY

    for i in range(len(hazard_cells)):
        for j in range(i + 1, len(hazard_cells)):
            a, b = hazard_cells[i], hazard_cells[j]
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            if d < MIN_SEPARATION:
                e_total += W_SEPARATION * (MIN_SEPARATION - d)

    from .validate import estimate_difficulty  # local import avoids a cycle
    delta = abs(estimate_difficulty(scenario, site) - scenario.difficulty_target)
    e_total += W_DIFFICULTY * delta

    return e_total


def random_placement(scenario: Scenario, site: SiteModel, rng: random.Random) -> Scenario:
    """Unconstrained baseline: drop every entity anywhere on the grid.

    This is the ablation arm that shows what the constraint solver is worth.
    """
    for ent in scenario.entities:
        ent.x = float(rng.randrange(site.width))
        ent.y = float(rng.randrange(site.depth))
        zone = site.zone_at(ent.x, ent.y)
        ent.zone = zone.name if zone else None
    return scenario


def solve(
    scenario: Scenario,
    site: Optional[SiteModel] = None,
    seed: int = 0,
    iterations: int = 900,
    t_start: float = 2.4,
    t_end: float = 0.02,
) -> Tuple[Scenario, float]:
    """Simulated annealing placement. Returns the scenario and its final energy."""
    site = site or SiteModel.load(scenario.site_template)
    rng = random.Random(seed)

    allowed: Dict[str, set] = {}
    candidates: Dict[str, List[Cell]] = {}
    for ent in scenario.entities:
        cells = _allowed_cells(ent, scenario, site)
        if not cells:
            cells = site.all_free_cells()
        candidates[ent.id] = cells
        allowed[ent.id] = set(cells)

    # Greedy, collision-aware initialisation.
    taken: set = set()
    for ent in scenario.entities:
        pool = [c for c in candidates[ent.id] if c not in taken] or candidates[ent.id]
        cell = rng.choice(pool)
        taken.add(cell)
        ent.x, ent.y = float(cell[0]), float(cell[1])

    current = energy(scenario, site, allowed)
    best = current
    best_state = [(e.id, e.x, e.y) for e in scenario.entities]

    for step in range(iterations):
        if current <= 0.0:
            break
        frac = step / max(iterations - 1, 1)
        temperature = t_start * (t_end / t_start) ** frac

        ent = rng.choice(scenario.entities)
        old = (ent.x, ent.y)
        if rng.random() < 0.35 and ent.placed:
            # local jitter keeps good layouts and repairs small clashes
            nx = int(ent.x) + rng.choice((-1, 0, 1))
            ny = int(ent.y) + rng.choice((-1, 0, 1))
            new_cell = (nx, ny)
            if new_cell not in allowed[ent.id]:
                new_cell = rng.choice(candidates[ent.id])
        else:
            new_cell = rng.choice(candidates[ent.id])
        ent.x, ent.y = float(new_cell[0]), float(new_cell[1])

        trial = energy(scenario, site, allowed)
        delta = trial - current
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-6)):
            current = trial
            if trial < best:
                best = trial
                best_state = [(e.id, e.x, e.y) for e in scenario.entities]
        else:
            ent.x, ent.y = old

    for entity_id, x, y in best_state:
        ent = scenario.entity(entity_id)
        ent.x, ent.y = x, y

    for ent in scenario.entities:
        zone = site.zone_at(ent.x, ent.y) if ent.placed else None
        ent.zone = zone.name if zone else None

    return scenario, best
