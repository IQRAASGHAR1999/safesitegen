"""Scenario graph synthesis.

The default backend is a stochastic grammar over the hazard taxonomy. It runs
offline, is deterministic under a seed, and produces exactly the same closed
vocabulary structure that a schema constrained language model would emit, so the
downstream solver, validator and exporter are backend agnostic. ``llm_backend``
holds an adapter for the language model path.

The grammar backend does not hallucinate. To measure what the validation gate is
actually worth, ``fault_rate`` injects specification noise of the kind a
generative model does produce: a control silently dropped, a parameter pushed
across a regulatory threshold, or a teaching point declared but never realised.
The gate is then scored on how much of that it catches.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .knowledge import DISTRACTORS, HAZARD_BY_ID, RULES_BY_ID, classes_for_zone, rule_for_class
from .layout import feasible_difficulty_range, random_placement, solve
from .schema import Entity, Hazard, Relation, Scenario
from .site import SiteModel
from .validate import ValidationReport, validate

TRADES_BY_SITE = {
    "steel_frame_level3": [
        ("ironworker", "steel decking and connection"),
        ("concrete_finisher", "deck pour preparation"),
        ("mason", "exterior blockwork at the perimeter"),
        ("electrician", "temporary power distribution on the deck"),
    ],
    "utility_trench_corridor": [
        ("pipelayer", "storm drain tie-in"),
        ("labourer", "bedding and backfill"),
        ("surveyor", "invert level checks"),
    ],
    "highway_workzone_taper": [
        ("highway_crew", "lane closure and milling"),
        ("flagger", "traffic control at the taper"),
        ("paving_crew", "overlay placement"),
    ],
}
DEFAULT_TRADES = [("labourer", "general site works")]

FAULT_KINDS = ("drop_control", "push_threshold", "phantom_hazard")


@dataclass
class GenerationResult:
    scenario: Scenario
    report: ValidationReport
    accepted: bool
    attempts: int
    energy: float
    injected_faults: List[str]
    repairs_applied: List[str]


# --------------------------------------------------------------------------- #
# Grammar backend
# --------------------------------------------------------------------------- #

def _sample_violating_value(hazard_class: str, rng: random.Random) -> Dict[str, float]:
    """Draw parameters that place the entity on the non-compliant side of the clause."""
    meta = HAZARD_BY_ID[hazard_class]
    out: Dict[str, float] = {}
    for param, bounds in meta.get("params", {}).items():
        lo, hi = bounds
        out[param] = round(rng.uniform(lo, hi), 1)
    return out


def _available_classes(site: SiteModel) -> List[str]:
    present = {z.type for z in site.zones}
    out: List[str] = []
    for zt in present:
        out.extend(classes_for_zone(zt))
    return sorted(set(out))


def build_scenario(
    site: SiteModel,
    target_classes: Sequence[str],
    difficulty_target: float,
    rng: random.Random,
    scenario_id: str,
    n_distractors: int = 3,
    trade: Optional[str] = None,
) -> Scenario:
    """Instantiate the semantic layer: entities, relations and teaching points."""
    options = TRADES_BY_SITE.get(site.id, DEFAULT_TRADES)
    if trade:
        match = next((t for t in options if t[0] == trade), None)
        trade, activity = match if match else (trade, "site works")
    else:
        trade, activity = rng.choice(options)
    scenario = Scenario(
        id=scenario_id,
        site_template=site.id,
        trade=trade,
        activity=activity,
        target_hazard_classes=list(target_classes),
        difficulty_target=difficulty_target,
        provenance={"backend": "grammar", "site_provenance": site.provenance},
    )

    counter = 0
    for hazard_class in target_classes:
        meta = HAZARD_BY_ID[hazard_class]
        rule = rule_for_class(hazard_class)
        context = rule.get("trigger", {}).get("context")

        victim_tpl = next(e for e in meta["entities"] if e["role"] == "victim")
        counter += 1
        victim_id = f"e{counter}_{hazard_class}"
        params = dict(victim_tpl.get("params", {}))
        params.update(_sample_violating_value(hazard_class, rng))
        if context:
            params["context"] = context
        params["controls"] = []
        victim = Entity(
            id=victim_id,
            kind=victim_tpl["kind"],
            asset_type=victim_tpl["asset_type"],
            params=params,
        )
        scenario.entities.append(victim)

        for tpl in meta["entities"]:
            if tpl["role"] == "victim":
                continue
            counter += 1
            ctx_id = f"e{counter}_ctx"
            ctx_entity = Entity(
                id=ctx_id,
                kind=tpl["kind"],
                asset_type=tpl["asset_type"],
                params=dict(tpl.get("params", {})),
            )
            scenario.entities.append(ctx_entity)
            scenario.relations.append(Relation(src=victim_id, dst=ctx_id, type="exposed_to"))

        scenario.hazards.append(
            Hazard(
                id=f"h{len(scenario.hazards) + 1}",
                hazard_class=hazard_class,
                target_entity=victim_id,
                clause=meta["clause"],
                energy_source=meta["energy_source"],
                required_controls=list(meta["required_controls"]),
                present_controls=[],
                severity="fatal" if meta["energy_source"] in ("gravitational", "electrical") else "serious",
                cue_salience=float(meta["cue_salience"]),
            )
        )

    for _ in range(n_distractors):
        tpl = rng.choice(DISTRACTORS)
        counter += 1
        scenario.entities.append(
            Entity(
                id=f"e{counter}_dist",
                kind=tpl["kind"],
                asset_type=tpl["asset_type"],
                params=dict(tpl.get("params", {})),
                is_distractor=True,
            )
        )

    return scenario


# --------------------------------------------------------------------------- #
# Hallucination model
# --------------------------------------------------------------------------- #

def inject_faults(scenario: Scenario, rate: float, rng: random.Random) -> List[str]:
    """Inject specification noise of the kind generative models produce."""
    injected: List[str] = []
    if rate <= 0.0:
        return injected

    for entity in list(scenario.entities):
        if rng.random() >= rate:
            continue
        kind = rng.choice(FAULT_KINDS)

        if kind == "drop_control" and entity.params.get("controls"):
            dropped = list(entity.params["controls"])
            entity.params["controls"] = []
            injected.append(f"drop_control:{entity.id}:{','.join(dropped)}")

        elif kind == "push_threshold":
            for param, shift in (
                ("top_rail_height_in", -12.0),
                ("extension_above_landing_ft", -3.0),
                ("distance_from_edge_ft", -4.0),
                ("clearance_to_line_ft", -15.0),
            ):
                if param in entity.params:
                    entity.params[param] = max(0.0, float(entity.params[param]) + shift)
                    injected.append(f"push_threshold:{entity.id}:{param}")
                    break

        elif kind == "phantom_hazard" and scenario.hazards:
            haz = rng.choice(scenario.hazards)
            target = scenario.entity(haz.target_entity)
            rule = RULES_BY_ID.get(haz.clause, {})
            controls_any = rule.get("predicate", {}).get("controls_any", [])
            if controls_any:
                target.params["controls"] = [controls_any[0]]
                haz.present_controls = [controls_any[0]]
                injected.append(f"phantom_hazard:{haz.id}")

    return injected


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #

def repair(scenario: Scenario, report: ValidationReport, rng: random.Random) -> List[str]:
    """Targeted repair driven by the failure ids, not a blind resample."""
    applied: List[str] = []
    declared = {h.target_entity for h in scenario.hazards}

    for check in report.failures():
        if check.check_id == "REG.no_unintended_violation" and check.entity:
            entity = scenario.entity(check.entity)
            rule = RULES_BY_ID.get(check.clause or "")
            if rule:
                pred = rule["predicate"]
                controls = pred.get("controls_any", [])
                if controls:
                    entity.params["controls"] = sorted(set(entity.params.get("controls", [])) | {controls[0]})
                if pred["kind"] == "range":
                    entity.params[pred["param"]] = (pred["min"] + pred["max"]) / 2.0
                elif pred["kind"] in ("threshold_or_control", "exclusion_zone"):
                    entity.params[pred["param"]] = pred["value"] + 1.0
                applied.append(f"restore_compliance:{entity.id}")

        elif check.check_id == "REG.hazard_realised" and check.entity in declared:
            entity = scenario.entity(check.entity)
            haz = next(h for h in scenario.hazards if h.target_entity == entity.id)
            entity.params["controls"] = []
            haz.present_controls = []
            entity.params.update(_sample_violating_value(haz.hazard_class, rng))
            applied.append(f"reassert_hazard:{haz.id}")

        elif check.check_id == "PED.hazard_count" and len(scenario.hazards) > 6:
            keep = scenario.hazards[:6]
            dropped = [h.id for h in scenario.hazards[6:]]
            scenario.hazards = keep
            applied.append(f"trim_hazards:{','.join(dropped)}")

        elif check.check_id == "PED.difficulty_band":
            # distractor pressure is the one difficulty term that can be changed
            # without moving a teaching point, so tune it before re-solving
            current = sum(1 for e in scenario.entities if e.is_distractor)
            want_harder = "away from the target" in check.detail and \
                float(check.detail.split("difficulty ")[1].split(" ")[0]) < scenario.difficulty_target
            if want_harder and current < 7:
                for _ in range(min(3, 7 - current)):
                    tpl = rng.choice(DISTRACTORS)
                    scenario.entities.append(Entity(
                        id=f"e{len(scenario.entities) + 1}_dist", kind=tpl["kind"],
                        asset_type=tpl["asset_type"], params=dict(tpl.get("params", {})),
                        is_distractor=True))
                applied.append("raise_distractor_pressure")
            elif not want_harder and current > 1:
                for e in list(scenario.entities):
                    if e.is_distractor and current > 1:
                        scenario.entities.remove(e)
                        current -= 1
                        if current <= 1:
                            break
                applied.append("lower_distractor_pressure")

        elif check.check_id == "PED.distractor_present":
            tpl = rng.choice(DISTRACTORS)
            scenario.entities.append(
                Entity(
                    id=f"e{len(scenario.entities) + 1}_dist",
                    kind=tpl["kind"],
                    asset_type=tpl["asset_type"],
                    params=dict(tpl.get("params", {})),
                    is_distractor=True,
                )
            )
            applied.append("add_distractor")

    return applied


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def generate(
    site_template: Optional[str] = None,
    target_classes: Optional[Sequence[str]] = None,
    difficulty_target: float = 0.5,
    seed: int = 0,
    n_hazards: Optional[int] = None,
    n_distractors: int = 3,
    trade: Optional[str] = None,
    fault_rate: float = 0.0,
    use_solver: bool = True,
    use_gate: bool = True,
    max_repairs: int = 3,
    scenario_id: Optional[str] = None,
) -> GenerationResult:
    """Full generate, realise, verify, repair pipeline.

    ``use_solver`` and ``use_gate`` exist so the ablation arms share one code
    path; turning both off gives the unconstrained baseline.

    ``n_hazards`` only tops up a partial class list when it is given explicitly,
    so naming two classes and saying nothing about a count delivers exactly two.
    """
    rng = random.Random(seed)
    site = SiteModel.load(site_template) if site_template else SiteModel.load(rng.choice(SiteModel.ids()))

    pool = _available_classes(site)
    if target_classes is None:
        target_classes = rng.sample(pool, min(n_hazards or 3, len(pool)))
    else:
        # named classes are always delivered; an explicit count tops the rest up
        target_classes = [c for c in target_classes if c in HAZARD_BY_ID]
        if n_hazards is not None and n_hazards > len(target_classes):
            spare = [c for c in pool if c not in target_classes]
            rng.shuffle(spare)
            target_classes = list(target_classes) + spare[: n_hazards - len(target_classes)]

    scenario_id = scenario_id or f"scn_{site.id}_{seed:05d}"
    scenario = build_scenario(site, target_classes, difficulty_target, rng, scenario_id,
                              n_distractors, trade=trade)

    # Clamp an impossible request to what the geometry can actually deliver,
    # and record that it was clamped rather than quietly missing the target.
    lo, hi = feasible_difficulty_range(scenario, site, seed=seed)
    clamped = None
    if not lo - 0.05 <= difficulty_target <= hi + 0.05:
        clamped = max(lo, min(hi, difficulty_target))
        scenario.difficulty_target = round(clamped, 3)
    injected = inject_faults(scenario, fault_rate, rng)

    if use_solver:
        scenario, e = solve(scenario, site, seed=seed)
    else:
        scenario = random_placement(scenario, site, rng)
        e = float("nan")

    report = validate(scenario, site)
    attempts = 1
    repairs: List[str] = []

    if use_gate:
        while not report.passed and attempts <= max_repairs:
            repairs.extend(repair(scenario, report, rng))
            # Re-place using the same strategy as the arm under test, so ablating
            # the solver really does ablate it everywhere.
            if use_solver:
                scenario, e = solve(scenario, site, seed=seed + attempts * 977, iterations=1200)
            else:
                scenario = random_placement(scenario, site, rng)
            report = validate(scenario, site)
            attempts += 1

    accepted = report.passed if use_gate else True
    scenario.seed = seed
    scenario.provenance.update(
        {
            "attempts": attempts,
            "injected_faults": injected,
            "repairs_applied": repairs,
            "use_solver": use_solver,
            "use_gate": use_gate,
            "difficulty_requested": difficulty_target,
            "difficulty_feasible_range": [lo, hi],
            "difficulty_clamped_to": round(clamped, 3) if clamped is not None else None,
        }
    )
    return GenerationResult(scenario, report, accepted, attempts, e, injected, repairs)
