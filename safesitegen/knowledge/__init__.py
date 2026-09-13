"""Knowledge sources: regulatory rule pack and hazard taxonomy."""

from __future__ import annotations

import json
from importlib import resources
from typing import Dict, List


def _load(name: str) -> dict:
    with resources.files(__name__).joinpath(name).open() as fh:
        return json.load(fh)


_RULES_DOC = _load("osha_1926.json")
_TAX_DOC = _load("hazard_taxonomy.json")

RULES: List[dict] = _RULES_DOC["rules"]
RULES_BY_ID: Dict[str, dict] = {r["id"]: r for r in RULES}
RULE_META: dict = _RULES_DOC["_meta"]

HAZARD_CLASSES: List[dict] = _TAX_DOC["classes"]
HAZARD_BY_ID: Dict[str, dict] = {h["id"]: h for h in HAZARD_CLASSES}
DISTRACTORS: List[dict] = _TAX_DOC["distractors"]
TAXONOMY_META: dict = _TAX_DOC["_meta"]


def rule_for_class(hazard_class: str) -> dict:
    """Return the regulatory rule a hazard class is defined against."""
    clause = HAZARD_BY_ID[hazard_class]["clause"]
    return RULES_BY_ID[clause]


def classes_for_zone(zone_type: str) -> List[str]:
    return [h["id"] for h in HAZARD_CLASSES if zone_type in h["zone_types"]]


__all__ = [
    "RULES",
    "RULES_BY_ID",
    "RULE_META",
    "HAZARD_CLASSES",
    "HAZARD_BY_ID",
    "DISTRACTORS",
    "TAXONOMY_META",
    "rule_for_class",
    "classes_for_zone",
]
