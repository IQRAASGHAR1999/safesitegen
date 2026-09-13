"""Optional language model backend for scenario graph synthesis.

The grammar backend in ``generator.py`` is the default because it runs offline
and deterministically, which is what the experiments need. This module is the
swap-in path for a constrained language model, and it exists mainly to pin down
the contract: the model never writes geometry and never writes free text, it
emits one JSON object against the schema below, and anything it produces still
goes through the same validation gate.

Nothing here is imported by the core pipeline, so the package has no runtime
dependency on a model or an API key.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from .knowledge import HAZARD_BY_ID, RULES_BY_ID
from .schema import Entity, Hazard, Scenario

SCENARIO_JSON_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["trade", "activity", "hazards", "distractors"],
    "properties": {
        "trade": {"type": "string"},
        "activity": {"type": "string", "maxLength": 120},
        "hazards": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hazard_class", "params", "rationale"],
                "properties": {
                    "hazard_class": {"type": "string", "enum": sorted(HAZARD_BY_ID)},
                    "params": {"type": "object"},
                    "present_controls": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string", "maxLength": 300},
                },
            },
        },
        "distractors": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string"},
        },
    },
}

SYSTEM_PROMPT = """You author construction safety training scenarios as structured data.

Rules you must follow:
1. Reply with one JSON object matching the supplied schema. No prose, no markdown.
2. Use only hazard_class values from the enum. Never invent a class or a clause.
3. Every hazard you declare must be genuinely non-compliant with its clause once
   the parameters you choose are applied. Do not declare a hazard and then supply
   a control that satisfies the clause.
4. Everything you do not declare as a hazard must be compliant. A trainee who
   spots an undeclared defect would be scored wrong, so undeclared defects are a
   defect in your output.
5. Ground each rationale in the retrieved incident narratives, not in invention.

You do not choose positions. A constraint solver places every entity afterwards.
"""


def build_prompt(
    target_classes: Sequence[str],
    difficulty_target: float,
    site_summary: str,
    retrieved_context: Sequence[str] = (),
) -> List[Dict[str, str]]:
    clauses = {c: HAZARD_BY_ID[c]["clause"] for c in target_classes}
    rule_text = "\n".join(
        f"- {c} -> {RULES_BY_ID[clauses[c]]['id']}: {RULES_BY_ID[clauses[c]]['summary']}"
        for c in target_classes
        if clauses[c] in RULES_BY_ID
    )
    context_block = "\n".join(f"- {c}" for c in retrieved_context) or "- (none retrieved)"
    user = (
        f"Site: {site_summary}\n"
        f"Target hazard classes: {', '.join(target_classes)}\n"
        f"Target difficulty (0 easy, 1 hard): {difficulty_target:.2f}\n\n"
        f"Governing clauses:\n{rule_text}\n\n"
        f"Retrieved incident narratives:\n{context_block}\n\n"
        f"Schema:\n{json.dumps(SCENARIO_JSON_SCHEMA)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class LLMBackend:
    """Thin adapter over any OpenAI-compatible chat completions endpoint.

    Constrained decoding (JSON schema or a context free grammar) should be
    enabled server side. Without it the parse will occasionally fail, which the
    caller must treat as a rejected sample rather than retry silently.
    """

    def __init__(self, model: str = "", base_url: str = "", api_key: Optional[str] = None):
        self.model = model or os.environ.get("SAFESITEGEN_MODEL", "")
        self.base_url = base_url or os.environ.get("SAFESITEGEN_BASE_URL", "")
        self.api_key = api_key or os.environ.get("SAFESITEGEN_API_KEY")

    @property
    def configured(self) -> bool:
        return bool(self.model and self.base_url and self.api_key)

    def complete(self, messages: List[Dict[str, str]], timeout: float = 60.0) -> str:
        if not self.configured:
            raise RuntimeError(
                "LLM backend is not configured. Set SAFESITEGEN_MODEL, "
                "SAFESITEGEN_BASE_URL and SAFESITEGEN_API_KEY, or use the grammar backend."
            )
        import urllib.request  # imported lazily so the package stays offline by default

        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.8,
                "response_format": {"type": "json_object"},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"]


def parse_scenario(raw: str, site_id: str, scenario_id: str, difficulty_target: float) -> Scenario:
    """Turn a model reply into an HSG. Raises on anything outside the vocabulary."""
    data = json.loads(raw)
    scenario = Scenario(
        id=scenario_id,
        site_template=site_id,
        trade=str(data.get("trade", "")),
        activity=str(data.get("activity", "")),
        difficulty_target=difficulty_target,
        provenance={"backend": "llm"},
    )
    for i, spec in enumerate(data["hazards"], start=1):
        cls = spec["hazard_class"]
        if cls not in HAZARD_BY_ID:
            raise ValueError(f"hazard class {cls!r} is outside the closed vocabulary")
        meta = HAZARD_BY_ID[cls]
        rule = RULES_BY_ID[meta["clause"]]
        victim = next(e for e in meta["entities"] if e["role"] == "victim")
        params = dict(victim.get("params", {}))
        params.update(spec.get("params", {}))
        if rule.get("trigger", {}).get("context"):
            params["context"] = rule["trigger"]["context"]
        params["controls"] = list(spec.get("present_controls", []))
        entity_id = f"e{i}_{cls}"
        scenario.entities.append(
            Entity(id=entity_id, kind=victim["kind"], asset_type=victim["asset_type"], params=params)
        )
        scenario.hazards.append(
            Hazard(
                id=f"h{i}",
                hazard_class=cls,
                target_entity=entity_id,
                clause=meta["clause"],
                energy_source=meta["energy_source"],
                required_controls=list(meta["required_controls"]),
                present_controls=list(spec.get("present_controls", [])),
                cue_salience=float(meta["cue_salience"]),
            )
        )
    scenario.target_hazard_classes = [h.hazard_class for h in scenario.hazards]
    return scenario
