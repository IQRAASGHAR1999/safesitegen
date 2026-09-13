"""The validation gate.

Games measure generated content against playability. Safety training content has
to clear a much harder bar, and the bar has two sides that are easy to conflate:

  1. every hazard the scenario *claims* to teach must actually be present in the
     realised geometry, otherwise the trainee is marked wrong for missing a
     hazard that was never there; and
  2. nothing the scenario does *not* claim may be non-compliant, otherwise the
     trainee is marked wrong for spotting a real hazard the generator invented
     by accident.

Both failure modes teach an incorrect mental model, which is worse than no
training at all. The regulatory layer below tests both directions explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from .knowledge import RULES, RULES_BY_ID, HAZARD_BY_ID
from .schema import Entity, Scenario
from .site import SiteModel

COMPLIANT = "compliant"
VIOLATION = "violation"
NOT_APPLICABLE = "not_applicable"

MIN_VISIBILITY = 0.15
MAX_HAZARDS = 6
MIN_HAZARDS = 1
DIFFICULTY_TOLERANCE = 0.28


@dataclass
class CheckResult:
    check_id: str
    layer: str
    status: str            # "pass" | "fail"
    detail: str
    clause: Optional[str] = None
    entity: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.status == "fail"


@dataclass
class ValidationReport:
    scenario_id: str
    checks: List[CheckResult] = field(default_factory=list)
    difficulty: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def passed(self) -> bool:
        return not any(c.failed for c in self.checks)

    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if c.failed]

    def failure_ids(self) -> List[str]:
        return [c.check_id for c in self.failures()]

    def by_layer(self, layer: str) -> List[CheckResult]:
        return [c for c in self.checks if c.layer == layer]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "difficulty": round(self.difficulty, 3),
            "metrics": self.metrics,
            "checks": [asdict(c) for c in self.checks],
        }

    def summary(self) -> str:
        if self.passed:
            return f"PASS  ({len(self.checks)} checks, difficulty {self.difficulty:.2f})"
        lines = [f"FAIL  ({len(self.failures())}/{len(self.checks)} checks failed)"]
        for c in self.failures():
            tag = f" [{c.clause}]" if c.clause else ""
            lines.append(f"  - {c.check_id}{tag}: {c.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Regulatory layer
# --------------------------------------------------------------------------- #

def _controls(entity: Entity) -> List[str]:
    return list(entity.params.get("controls", []))


def _rule_applies(rule: dict, entity: Entity) -> bool:
    trig = rule.get("trigger", {})
    if "asset_type" in trig:
        return entity.asset_type == trig["asset_type"]
    if "entity_kind" in trig:
        if entity.kind != trig["entity_kind"]:
            return False
        ctx = trig.get("context")
        if ctx is None:
            return True
        return entity.params.get("context") == ctx
    return False


def _cmp(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    raise ValueError(f"unsupported operator {op!r}")


def evaluate_rule(rule: dict, entity: Entity) -> str:
    """Evaluate one clause against one entity.

    Returns COMPLIANT, VIOLATION or NOT_APPLICABLE. Deterministic and
    side-effect free, so every verdict is traceable back to a clause id.
    """
    if not _rule_applies(rule, entity):
        return NOT_APPLICABLE

    pred = rule["predicate"]
    kind = pred["kind"]
    controls = _controls(entity)

    exempt = pred.get("exempt_if")
    if exempt and entity.params.get(exempt["param"]) == exempt["equals"]:
        return COMPLIANT

    if kind == "requires_any_control":
        ok = any(c in controls for c in pred["controls_any"])
        return COMPLIANT if ok else VIOLATION

    if kind == "range":
        value = entity.params.get(pred["param"])
        if value is None:
            return NOT_APPLICABLE
        ok = pred["min"] <= float(value) <= pred["max"]
        return COMPLIANT if ok else VIOLATION

    if kind == "threshold_requires_any_control":
        value = entity.params.get(pred["param"])
        if value is None:
            return NOT_APPLICABLE
        if not _cmp(float(value), pred["op"], pred["value"]):
            return COMPLIANT          # below the regulatory trigger height
        ok = any(c in controls for c in pred["controls_any"])
        return COMPLIANT if ok else VIOLATION

    if kind in ("threshold_or_control", "exclusion_zone"):
        value = entity.params.get(pred["param"])
        if value is None:
            return NOT_APPLICABLE
        if _cmp(float(value), pred["op"], pred["value"]):
            return COMPLIANT
        ok = any(c in controls for c in pred.get("controls_any", []))
        return COMPLIANT if ok else VIOLATION

    if kind == "lateral_travel":
        depth = entity.params.get(pred["depth_param"])
        travel = entity.params.get(pred["param"])
        if depth is None or travel is None:
            return NOT_APPLICABLE
        if float(depth) < pred["depth_min"]:
            return COMPLIANT
        return COMPLIANT if _cmp(float(travel), pred["op"], pred["value"]) else VIOLATION

    raise ValueError(f"unsupported predicate kind {kind!r}")


def audit(scenario: Scenario) -> Dict[str, List[str]]:
    """Full clause-by-entity audit. Returns {entity_id: [violated clause ids]}."""
    out: Dict[str, List[str]] = {}
    for entity in scenario.entities:
        hits = [r["id"] for r in RULES if evaluate_rule(r, entity) == VIOLATION]
        if hits:
            out[entity.id] = hits
    return out


def check_regulatory(scenario: Scenario, report: ValidationReport) -> None:
    violations = audit(scenario)
    intended: Dict[str, set] = {}
    for h in scenario.hazards:
        intended.setdefault(h.target_entity, set()).add(h.clause)

    # 1. Every declared hazard must actually be realised.
    for h in scenario.hazards:
        rule = RULES_BY_ID.get(h.clause)
        entity = scenario.entity(h.target_entity)
        if rule is None:
            report.add(CheckResult(
                "REG.unknown_clause", "regulatory", "fail",
                f"hazard {h.id} cites clause {h.clause} which is not in the rule pack",
                clause=h.clause, entity=entity.id))
            continue
        verdict = evaluate_rule(rule, entity)
        if verdict == VIOLATION:
            report.add(CheckResult(
                "REG.hazard_realised", "regulatory", "pass",
                f"{h.hazard_class} is present on {entity.id} as declared",
                clause=h.clause, entity=entity.id))
        else:
            report.add(CheckResult(
                "REG.hazard_realised", "regulatory", "fail",
                f"hazard {h.id} claims a {h.clause} violation on {entity.id} "
                f"but the realised configuration evaluates as {verdict}",
                clause=h.clause, entity=entity.id))

    # 2. Nothing else may be non-compliant.
    for entity_id, clauses in violations.items():
        unintended = [c for c in clauses if c not in intended.get(entity_id, set())]
        if unintended:
            report.add(CheckResult(
                "REG.no_unintended_violation", "regulatory", "fail",
                f"entity {entity_id} violates {', '.join(unintended)} without a matching "
                f"teaching point, so a correct trainee answer would be scored wrong",
                clause=unintended[0], entity=entity_id))

    if not any(c.check_id == "REG.no_unintended_violation" for c in report.checks):
        report.add(CheckResult(
            "REG.no_unintended_violation", "regulatory", "pass",
            "no unlabelled regulatory violations in the scene"))


# --------------------------------------------------------------------------- #
# Physical layer
# --------------------------------------------------------------------------- #

def check_physical(scenario: Scenario, site: SiteModel, report: ValidationReport) -> None:
    occupied: Dict[tuple, str] = {}
    reachable = site.reachable_cells()

    for e in scenario.entities:
        if not e.placed:
            report.add(CheckResult(
                "PHY.placed", "physical", "fail",
                f"entity {e.id} was never placed", entity=e.id))
            continue

        cell = (int(e.x), int(e.y))

        if not site.in_bounds(*cell):
            report.add(CheckResult(
                "PHY.in_bounds", "physical", "fail",
                f"entity {e.id} at {cell} lies outside the site substrate", entity=e.id))
            continue

        if site.is_blocked(*cell):
            report.add(CheckResult(
                "PHY.no_interpenetration", "physical", "fail",
                f"entity {e.id} intersects fixed site geometry at {cell}", entity=e.id))

        if cell in occupied:
            report.add(CheckResult(
                "PHY.no_overlap", "physical", "fail",
                f"entities {occupied[cell]} and {e.id} occupy the same cell {cell}",
                entity=e.id))
        else:
            occupied[cell] = e.id

        if site.zone_at(*cell) is None:
            report.add(CheckResult(
                "PHY.supported", "physical", "fail",
                f"entity {e.id} at {cell} is not on any defined working surface", entity=e.id))

        if e.kind in ("worker", "equipment") and cell not in reachable:
            report.add(CheckResult(
                "PHY.reachable", "physical", "fail",
                f"{e.kind} {e.id} at {cell} cannot be reached from the site entry point",
                entity=e.id))

    # A cave-in exposure needs an excavation under it; a leading-edge fall needs an
    # edge. Placing a teaching point where its affordance does not exist produces a
    # scene that is regulatorily "correct" and physically impossible.
    for h in scenario.hazards:
        meta = HAZARD_BY_ID.get(h.hazard_class)
        if not meta:
            continue
        e = scenario.entity(h.target_entity)
        if not e.placed:
            continue
        zone = site.zone_at(int(e.x), int(e.y))
        if zone is None or zone.type not in meta["zone_types"]:
            found = zone.type if zone else "open ground"
            report.add(CheckResult(
                "PHY.zone_affordance", "physical", "fail",
                f"{h.hazard_class} requires one of {', '.join(meta['zone_types'])} "
                f"but {e.id} sits on {found}", entity=e.id))

    for cid in ("PHY.placed", "PHY.in_bounds", "PHY.no_overlap", "PHY.supported",
                "PHY.reachable", "PHY.no_interpenetration", "PHY.zone_affordance"):
        if not any(c.check_id == cid for c in report.checks):
            report.add(CheckResult(cid, "physical", "pass", "satisfied for all entities"))


# --------------------------------------------------------------------------- #
# Pedagogical layer
# --------------------------------------------------------------------------- #

def estimate_difficulty(scenario: Scenario, site: SiteModel) -> float:
    """Heuristic difficulty in [0, 1] for expressive range analysis and targeting.

    Combines how hard each teaching point is to see, how far it sits from the
    trainee's route, how salient its perceptual cue is, how rare the class is,
    and how many safe-but-salient distractors compete for attention.
    """
    hazard_entities = [scenario.entity(h.target_entity) for h in scenario.hazards]
    if not hazard_entities:
        return 1.0

    vis, dist, salience, rarity = [], [], [], []
    for h, e in zip(scenario.hazards, hazard_entities):
        if not e.placed:
            vis.append(0.0)
            dist.append(1.0)
        else:
            cell = (int(e.x), int(e.y))
            vis.append(site.visibility_fraction(cell))
            span = max(site.width, site.depth)
            dist.append(min(site.nearest_patrol_distance(cell) / span, 1.0))
        salience.append(h.cue_salience)
        meta = HAZARD_BY_ID.get(h.hazard_class, {})
        rarity.append(1.0 - float(meta.get("base_rate", 0.1)) / 0.25)

    n_dist = sum(1 for e in scenario.entities if e.is_distractor)
    distractor_pressure = min(n_dist / 6.0, 1.0)

    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    score = (
        0.34 * (1.0 - mean(vis))
        + 0.16 * mean(dist)
        + 0.24 * (1.0 - mean(salience))
        + 0.14 * max(0.0, mean(rarity))
        + 0.12 * distractor_pressure
    )
    return max(0.0, min(1.0, score))


def check_pedagogical(scenario: Scenario, site: SiteModel, report: ValidationReport) -> None:
    n_haz = len(scenario.hazards)

    if n_haz < MIN_HAZARDS:
        report.add(CheckResult("PED.hazard_count", "pedagogical", "fail",
                               "scenario contains no teaching point"))
    elif n_haz > MAX_HAZARDS:
        report.add(CheckResult("PED.hazard_count", "pedagogical", "fail",
                               f"{n_haz} teaching points exceeds the working memory budget of {MAX_HAZARDS}"))
    else:
        report.add(CheckResult("PED.hazard_count", "pedagogical", "pass",
                               f"{n_haz} teaching points"))

    # Every hazard meant to be found must be perceivable from the trainee route.
    invisible = []
    for h in scenario.hazards:
        if not h.intended_detectable:
            continue
        e = scenario.entity(h.target_entity)
        if not e.placed:
            invisible.append((h.id, 0.0))
            continue
        frac = site.visibility_fraction((int(e.x), int(e.y)))
        if frac < MIN_VISIBILITY:
            invisible.append((h.id, frac))
    if invisible:
        detail = "; ".join(f"{hid} visible from {frac:.0%} of viewpoints" for hid, frac in invisible)
        report.add(CheckResult("PED.cue_perceivable", "pedagogical", "fail",
                               f"teaching points cannot be seen from the trainee route: {detail}"))
    else:
        report.add(CheckResult("PED.cue_perceivable", "pedagogical", "pass",
                               "every teaching point is visible from the trainee route"))

    # Distractors stop "anything unusual is a hazard" from being a winning strategy.
    n_distractors = sum(1 for e in scenario.entities if e.is_distractor)
    if n_distractors < 1:
        report.add(CheckResult("PED.distractor_present", "pedagogical", "fail",
                               "no safe-but-salient distractor, so guessing is rewarded"))
    else:
        report.add(CheckResult("PED.distractor_present", "pedagogical", "pass",
                               f"{n_distractors} distractors"))

    # No two teaching points stacked on one cell.
    cells: Dict[tuple, str] = {}
    clash = None
    for h in scenario.hazards:
        e = scenario.entity(h.target_entity)
        if not e.placed:
            continue
        cell = (int(e.x), int(e.y))
        if cell in cells:
            clash = (cells[cell], h.id, cell)
            break
        cells[cell] = h.id
    if clash:
        report.add(CheckResult("PED.hazard_separation", "pedagogical", "fail",
                               f"teaching points {clash[0]} and {clash[1]} coincide at {clash[2]}"))
    else:
        report.add(CheckResult("PED.hazard_separation", "pedagogical", "pass",
                               "teaching points are spatially separated"))

    # Requested curriculum must actually be delivered.
    realised = {h.hazard_class for h in scenario.hazards}
    missing = [c for c in scenario.target_hazard_classes if c not in realised]
    if missing:
        report.add(CheckResult("PED.curriculum_coverage", "pedagogical", "fail",
                               f"requested classes not delivered: {', '.join(missing)}"))
    else:
        report.add(CheckResult("PED.curriculum_coverage", "pedagogical", "pass",
                               "all requested hazard classes delivered"))

    difficulty = estimate_difficulty(scenario, site)
    report.difficulty = difficulty
    delta = abs(difficulty - scenario.difficulty_target)
    if delta > DIFFICULTY_TOLERANCE:
        report.add(CheckResult("PED.difficulty_band", "pedagogical", "fail",
                               f"difficulty {difficulty:.2f} is {delta:.2f} away from the "
                               f"target {scenario.difficulty_target:.2f}"))
    else:
        report.add(CheckResult("PED.difficulty_band", "pedagogical", "pass",
                               f"difficulty {difficulty:.2f} within band of "
                               f"{scenario.difficulty_target:.2f}"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def validate(scenario: Scenario, site: Optional[SiteModel] = None) -> ValidationReport:
    """Run all three checker layers and return an auditable report."""
    site = site or SiteModel.load(scenario.site_template)
    report = ValidationReport(scenario_id=scenario.id)

    try:
        scenario.validate()
    except Exception as exc:  # schema breach short-circuits the rest
        report.add(CheckResult("SCH.wellformed", "schema", "fail", str(exc)))
        return report
    report.add(CheckResult("SCH.wellformed", "schema", "pass", "scenario graph is well formed"))

    check_regulatory(scenario, report)
    check_physical(scenario, site, report)
    check_pedagogical(scenario, site, report)

    report.metrics = {
        "n_entities": len(scenario.entities),
        "n_hazards": len(scenario.hazards),
        "n_distractors": sum(1 for e in scenario.entities if e.is_distractor),
        "hazard_classes": sorted({h.hazard_class for h in scenario.hazards}),
        "signature": scenario.configuration_signature(),
    }
    return report
