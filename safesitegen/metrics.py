"""Generator-level metrics.

Validity rate answers "is the content usable". Expressive range analysis answers
"what part of the design space can this generator actually reach", which is the
question that exposes a generator quietly collapsing onto one kind of scenario.
The 2D binned view follows Smith and Whitehead's expressive range method,
substituting hazard diversity and difficulty for their platformer heuristics.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import Scenario


def validity_rate(passed_flags: Sequence[bool]) -> float:
    return sum(1 for p in passed_flags if p) / len(passed_flags) if passed_flags else 0.0


def hazard_diversity(scenario: Scenario) -> float:
    """Distinct hazard classes divided by teaching points; 1.0 means no repeats."""
    if not scenario.hazards:
        return 0.0
    return len({h.hazard_class for h in scenario.hazards}) / len(scenario.hazards)


def spatial_dispersion(scenario: Scenario, site) -> float:
    """Mean pairwise separation of teaching points, normalised by the site diagonal.

    Used as the second expressive-range axis because it is a genuine design
    dimension (are the hazards clustered in one bay or spread across the site)
    and is largely independent of the difficulty heuristic.
    """
    import math
    cells = []
    for h in scenario.hazards:
        e = scenario.entity(h.target_entity)
        if e.placed:
            cells.append((float(e.x), float(e.y)))
    if len(cells) < 2:
        return 0.0
    diag = math.hypot(site.width, site.depth) or 1.0
    dists = [
        math.hypot(cells[i][0] - cells[j][0], cells[i][1] - cells[j][1])
        for i in range(len(cells))
        for j in range(i + 1, len(cells))
    ]
    return min(1.0, (sum(dists) / len(dists)) / diag)


def energy_source_spread(scenario: Scenario) -> int:
    return len({h.energy_source for h in scenario.hazards})


def class_coverage(scenarios: Sequence[Scenario], universe: Sequence[str]) -> float:
    """Share of the taxonomy the generator ever reaches."""
    seen = {h.hazard_class for s in scenarios for h in s.hazards}
    return len(seen & set(universe)) / len(universe) if universe else 0.0


def class_entropy(scenarios: Sequence[Scenario]) -> float:
    """Normalised Shannon entropy of the hazard class distribution."""
    import math
    counts = Counter(h.hazard_class for s in scenarios for h in s.hazards)
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values())
    return h / math.log(len(counts))


def signature_uniqueness(scenarios: Sequence[Scenario]) -> float:
    sigs = [s.configuration_signature() for s in scenarios]
    return len(set(sigs)) / len(sigs) if sigs else 0.0


def expressive_range(
    points: Sequence[Tuple[float, float]],
    bins: int = 10,
) -> Dict[str, object]:
    """Bin (difficulty, diversity) pairs into a grid and report occupancy.

    ``coverage`` is the share of cells the generator ever visits, which is the
    headline expressive range number; ``peak_share`` exposes mode collapse.
    """
    grid = [[0 for _ in range(bins)] for _ in range(bins)]
    for x, y in points:
        ix = min(bins - 1, max(0, int(x * bins)))
        iy = min(bins - 1, max(0, int(y * bins)))
        grid[iy][ix] += 1

    occupied = sum(1 for row in grid for c in row if c > 0)
    total = sum(sum(row) for row in grid)
    peak = max((c for row in grid for c in row), default=0)
    return {
        "bins": bins,
        "grid": grid,
        "coverage": occupied / (bins * bins),
        "peak_share": peak / total if total else 0.0,
        "n": total,
    }


def summarise(
    scenarios: Sequence[Scenario],
    difficulties: Sequence[float],
    passed_flags: Sequence[bool],
    universe: Sequence[str],
    sites: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    diversities = [hazard_diversity(s) for s in scenarios]
    if sites is not None:
        dispersions = [spatial_dispersion(s, sites[s.site_template]) for s in scenarios]
    else:
        dispersions = diversities
    return {
        "n": len(scenarios),
        "validity_rate": round(validity_rate(passed_flags), 4),
        "class_coverage": round(class_coverage(scenarios, universe), 4),
        "class_entropy": round(class_entropy(scenarios), 4),
        "signature_uniqueness": round(signature_uniqueness(scenarios), 4),
        "difficulty_mean": round(mean(difficulties), 4) if difficulties else 0.0,
        "difficulty_sd": round(pstdev(difficulties), 4) if len(difficulties) > 1 else 0.0,
        "hazard_diversity_mean": round(mean(diversities), 4) if diversities else 0.0,
        "expressive_range": expressive_range(list(zip(difficulties, dispersions))),
    }


def svg_expressive_range(era: Dict[str, object], title: str = "Expressive range") -> str:
    """Dependency-free SVG heatmap so results render on GitHub without a plot library."""
    bins = int(era["bins"])  # type: ignore[arg-type]
    grid: List[List[int]] = era["grid"]  # type: ignore[assignment]
    peak = max((c for row in grid for c in row), default=1) or 1
    cell, pad = 26, 52
    w = h = bins * cell + pad + 16

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad}" y="20" font-family="Helvetica,Arial" font-size="13" fill="#111">{title}</text>',
    ]
    for iy in range(bins):
        for ix in range(bins):
            v = grid[iy][ix] / peak
            shade = int(255 - 205 * v)
            fill = "#ffffff" if v == 0 else f"rgb({shade},{min(255, shade + 26)},{255 - int(60 * v)})"
            x = pad + ix * cell
            y = 32 + (bins - 1 - iy) * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" '
                f'fill="{fill}" stroke="#dfe3e8" stroke-width="0.5"/>'
            )
    parts.append(
        f'<text x="{pad}" y="{32 + bins * cell + 16}" font-family="Helvetica,Arial" '
        f'font-size="11" fill="#444">difficulty →</text>'
    )
    parts.append(
        f'<text x="14" y="{32 + bins * cell}" font-family="Helvetica,Arial" font-size="11" '
        f'fill="#444" transform="rotate(-90 14 {32 + bins * cell})">spatial dispersion →</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
