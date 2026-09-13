"""Typed intermediate representation for construction safety training scenarios.

The Hazard Scenario Graph (HSG) is the pivot of the whole framework. It is
deliberately small, typed and closed vocabulary so that it can be

  * emitted by a constrained generator (grammar or schema constrained LLM),
  * checked symbolically against a regulatory rule pack,
  * realised geometrically by a placement solver, and
  * compiled into a runtime scene description.

Every field that a rule predicate can read lives in ``Entity.params``, which
keeps the checker independent of the asset catalogue.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

ENTITY_KINDS = frozenset({"worker", "equipment", "structure", "control", "environment"})

RELATION_TYPES = frozenset(
    {
        "located_in",       # entity -> zone
        "exposed_to",       # worker -> hazard context
        "protects",         # control -> entity
        "operates",         # worker -> equipment
        "supports",         # structure -> entity
        "adjacent_to",      # spatial proximity
        "occludes",         # entity blocks sightline to entity
        "precedes",         # temporal ordering of scenario beats
    }
)

SEVERITIES = ("minor", "serious", "fatal")


class SchemaError(ValueError):
    """Raised when a scenario violates the closed vocabulary or required fields."""


@dataclass
class Entity:
    id: str
    kind: str
    asset_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    x: Optional[float] = None
    y: Optional[float] = None
    level_ft: float = 0.0
    zone: Optional[str] = None
    is_distractor: bool = False

    @property
    def placed(self) -> bool:
        return self.x is not None and self.y is not None

    def position(self) -> Tuple[float, float]:
        if not self.placed:
            raise SchemaError(f"entity {self.id} has no position")
        return (float(self.x), float(self.y))

    def validate(self) -> None:
        if self.kind not in ENTITY_KINDS:
            raise SchemaError(f"entity {self.id}: unknown kind {self.kind!r}")
        if not self.asset_type:
            raise SchemaError(f"entity {self.id}: empty asset_type")


@dataclass
class Relation:
    src: str
    dst: str
    type: str

    def validate(self, entity_ids: frozenset) -> None:
        if self.type not in RELATION_TYPES:
            raise SchemaError(f"unknown relation type {self.type!r}")
        for endpoint in (self.src, self.dst):
            if endpoint not in entity_ids:
                raise SchemaError(f"relation {self.type} references unknown entity {endpoint!r}")


@dataclass
class Hazard:
    """An intended teaching point.

    ``clause`` names the regulatory requirement the scenario deliberately
    violates. ``present_controls`` is what the scene actually contains, which is
    normally a strict subset of ``required_controls``: the difference is the
    hazard.
    """

    id: str
    hazard_class: str
    target_entity: str
    clause: str
    energy_source: str
    required_controls: List[str] = field(default_factory=list)
    present_controls: List[str] = field(default_factory=list)
    severity: str = "serious"
    cue_salience: float = 0.5
    intended_detectable: bool = True

    def validate(self, entity_ids: frozenset) -> None:
        if self.target_entity not in entity_ids:
            raise SchemaError(f"hazard {self.id}: unknown target entity {self.target_entity!r}")
        if self.severity not in SEVERITIES:
            raise SchemaError(f"hazard {self.id}: unknown severity {self.severity!r}")
        if not 0.0 <= self.cue_salience <= 1.0:
            raise SchemaError(f"hazard {self.id}: cue_salience out of range")
        missing = set(self.present_controls) - set(self.required_controls)
        if missing and self.required_controls:
            raise SchemaError(
                f"hazard {self.id}: present controls {sorted(missing)} are not in the required set"
            )


@dataclass
class Scenario:
    id: str
    site_template: str
    trade: str
    activity: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    hazards: List[Hazard] = field(default_factory=list)
    target_hazard_classes: List[str] = field(default_factory=list)
    difficulty_target: float = 0.5
    seed: Optional[int] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- lookups
    def entity(self, entity_id: str) -> Entity:
        for e in self.entities:
            if e.id == entity_id:
                return e
        raise SchemaError(f"no entity {entity_id!r}")

    def entity_ids(self) -> frozenset:
        return frozenset(e.id for e in self.entities)

    def placed_entities(self) -> List[Entity]:
        return [e for e in self.entities if e.placed]

    def hazard_entities(self) -> List[Entity]:
        return [self.entity(h.target_entity) for h in self.hazards]

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        if not self.entities:
            raise SchemaError("scenario has no entities")
        seen = set()
        for e in self.entities:
            if e.id in seen:
                raise SchemaError(f"duplicate entity id {e.id!r}")
            seen.add(e.id)
            e.validate()
        ids = self.entity_ids()
        for r in self.relations:
            r.validate(ids)
        haz_ids = set()
        for h in self.hazards:
            if h.id in haz_ids:
                raise SchemaError(f"duplicate hazard id {h.id!r}")
            haz_ids.add(h.id)
            h.validate(ids)
        if not 0.0 <= self.difficulty_target <= 1.0:
            raise SchemaError("difficulty_target out of range")

    # -------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        return cls(
            id=data["id"],
            site_template=data["site_template"],
            trade=data.get("trade", ""),
            activity=data.get("activity", ""),
            entities=[Entity(**e) for e in data.get("entities", [])],
            relations=[Relation(**r) for r in data.get("relations", [])],
            hazards=[Hazard(**h) for h in data.get("hazards", [])],
            target_hazard_classes=data.get("target_hazard_classes", []),
            difficulty_target=data.get("difficulty_target", 0.5),
            seed=data.get("seed"),
            provenance=data.get("provenance", {}),
        )

    # ------------------------------------------------------------- signature
    def configuration_signature(self) -> str:
        """Layout-sensitive fingerprint used for the novelty constraint.

        Two scenarios that teach the same hazard classes in the same places
        collide here, which is exactly what we want to avoid showing a trainee
        twice.
        """
        parts = [self.site_template]
        for h in sorted(self.hazards, key=lambda z: z.id):
            e = self.entity(h.target_entity)
            cell = (int(e.x), int(e.y)) if e.placed else (-1, -1)
            parts.append(f"{h.hazard_class}@{cell[0]},{cell[1]}")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
