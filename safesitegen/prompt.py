"""Natural language front end.

A trainer does not want to type ``--classes trench_no_protective_system``. They
want to say "a storm drain crew working in an unshored trench with the spoil
piled on the edge, make it hard". This module turns that into a specification
the generator can act on, and turns a follow-up like "now add a crane near the
power line and make it harder" into a modification of the scene already on
screen.

Two backends, same output type:

  * ``lexicon`` (default) matches phrases against the closed vocabulary. It is
    deterministic, runs offline, and cannot invent a hazard class that does not
    exist, because it can only ever return identifiers that are already in the
    taxonomy.
  * ``llm`` routes the same request through a schema-constrained model via
    ``llm_backend``, for phrasing the lexicon misses.

Both produce a ``PromptSpec`` carrying a parse trace, so a demonstration can
show exactly which words drove which decision rather than asking an audience to
trust the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .knowledge import HAZARD_BY_ID
from .schema import Scenario
from .site import SiteModel

# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #

HAZARD_PHRASES: Dict[str, Sequence[str]] = {
    "fall_unprotected_edge": (
        "unprotected edge", "leading edge", "open edge", "slab edge", "deck edge",
        "no edge protection", "missing guardrail at the edge", "fall from the edge",
        "unguarded edge", "perimeter protection", "edge of the slab", "no handrail",
        "fall off the side", "working at height",
    ),
    "fall_floor_opening": (
        "floor opening", "floor hole", "uncovered hole", "open penetration",
        "uncovered opening", "hole in the deck", "unguarded opening", "shaft opening",
    ),
    "defective_guardrail": (
        "low guardrail", "defective guardrail", "guardrail too low", "short guardrail",
        "substandard guardrail", "top rail height", "undersized rail",
    ),
    "ladder_insufficient_extension": (
        "ladder", "extension ladder", "ladder not extended", "short ladder",
        "ladder access", "ladder above the landing", "climbing access",
    ),
    "scaffold_missing_guardrail": (
        "scaffold", "scaffolding", "scaffold platform", "staging", "no guardrail on the scaffold",
    ),
    "trench_no_protective_system": (
        "cave-in", "cave in", "unshored", "no shoring", "trench box", "protective system",
        "collapse", "unprotected trench", "no trench protection", "shoring",
    ),
    "trench_blocked_egress": (
        "egress", "no ladder in the trench", "exit from the trench", "escape route",
        "means of egress", "no way out",
    ),
    "trench_spoil_too_close": (
        "spoil", "spoil pile", "excavated material", "dirt pile", "surcharge",
        "material on the edge", "piled on the edge",
    ),
    "power_line_encroachment": (
        "power line", "powerline", "overhead line", "energised line", "energized line",
        "electrical line", "crane near the line", "boom near the line", "electrocution",
        "live conductor", "overhead cable", "high voltage",
    ),
    "struck_by_suspended_load": (
        "suspended load", "under the load", "fall zone", "load swinging", "swinging load",
        "swinging a load", "crane load", "rigging", "lifting operation", "lifting a load",
        "load overhead", "crane lift", "hoisting", "crane",
    ),
    "backing_equipment_no_spotter": (
        "backing", "reversing", "no spotter", "reverse alarm", "backup alarm",
        "truck reversing", "blind spot", "struck by vehicle", "backing up", "dump truck",
        "plant movement", "no banksman",
    ),
    "missing_head_protection": (
        "hard hat", "hardhat", "no helmet", "head protection", "ppe", "no hat",
    ),
    "rebar_impalement": (
        "rebar", "reinforcing steel", "impalement", "rebar caps", "protruding steel",
        "starter bars", "dowels",
    ),
    "workzone_traffic_exposure": (
        "live traffic", "traffic exposure", "work zone traffic", "flagger", "no barrier",
        "highway traffic", "passing vehicles", "positive protection",
    ),
}

SITE_PHRASES: Dict[str, Sequence[str]] = {
    "steel_frame_level3": (
        "steel frame", "steel building", "high rise", "highrise", "deck", "slab",
        "level 3", "third floor", "upper floor", "building", "structure", "ironwork",
        "concrete pour", "decking",
    ),
    "utility_trench_corridor": (
        "trench", "excavation", "utility", "storm drain", "sewer", "pipe", "pipeline",
        "underground", "dig", "digging", "drainage",
    ),
    "highway_workzone_taper": (
        "highway", "work zone", "workzone", "road", "roadway", "lane closure", "taper",
        "paving", "milling", "traffic", "street", "motorway",
    ),
}

DIFFICULTY_PHRASES: Sequence[Tuple[float, Sequence[str]]] = (
    (0.20, ("very easy", "beginner", "brand new", "first day", "induction", "simple", "obvious")),
    (0.32, ("easy", "novice", "new hire", "apprentice", "entry level", "gentle", "straightforward")),
    (0.50, ("moderate", "medium", "average", "standard", "typical", "normal")),
    (0.70, ("hard", "difficult", "challenging", "experienced", "advanced", "subtle", "tricky")),
    (0.85, ("very hard", "expert", "very difficult", "hardest", "brutal", "competent person")),
)

COUNT_WORDS = {
    "one": 1, "a single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "couple": 2, "pair": 2,
}

TRADE_PHRASES: Dict[str, Sequence[str]] = {
    "ironworker": ("ironworker", "steel erector", "steel crew", "connector"),
    "concrete_finisher": ("concrete", "finisher", "pour crew"),
    "pipelayer": ("pipelayer", "pipe crew", "drainage crew", "storm drain crew"),
    "mason": ("mason", "bricklayer", "blockwork"),
    "highway_crew": ("highway crew", "paving crew", "road crew"),
    "flagger": ("flagger", "traffic controller", "banksman"),
    "electrician": ("electrician", "sparky", "electrical crew"),
    "labourer": ("labourer", "laborer", "general hand"),
}

MODIFY_VERBS = ("add", "include", "put", "insert", "also", "now", "remove", "drop",
                "delete", "take out", "without", "replace", "swap", "change", "make it",
                "instead", "harder", "easier", "another")

REMOVE_CUES = ("remove", "drop", "delete", "take out", "without", "get rid of", "no more")

HARDER_CUES = ("harder", "more difficult", "more challenging", "tougher", "raise the difficulty",
               "step it up", "more subtle")
EASIER_CUES = ("easier", "less difficult", "simpler", "lower the difficulty", "more obvious",
               "more visible")


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class PromptSpec:
    """A parsed request. ``intent`` is either ``create`` or ``modify``."""

    intent: str = "create"
    text: str = ""
    site_template: Optional[str] = None
    target_classes: List[str] = field(default_factory=list)
    remove_classes: List[str] = field(default_factory=list)
    difficulty_target: Optional[float] = None
    difficulty_delta: float = 0.0
    n_hazards: Optional[int] = None
    trade: Optional[str] = None
    trace: List[str] = field(default_factory=list)
    backend: str = "lexicon"

    def note(self, message: str) -> None:
        self.trace.append(message)

    def explain(self) -> str:
        lines = [f'prompt   "{self.text}"', f"intent   {self.intent}  (backend: {self.backend})"]
        for t in self.trace:
            lines.append(f"  - {t}")
        return "\n".join(lines)

    def resolved(self) -> Dict[str, object]:
        return {
            "site_template": self.site_template,
            "target_classes": list(self.target_classes),
            "difficulty_target": self.difficulty_target,
            "n_hazards": self.n_hazards,
        }


# --------------------------------------------------------------------------- #
# Lexicon backend
# --------------------------------------------------------------------------- #

def _normalise(text: str) -> str:
    t = text.lower()
    t = t.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def _find_phrases(text: str, table: Dict[str, Sequence[str]]) -> List[Tuple[str, str]]:
    """Return (identifier, matched phrase), longest phrase first so specific beats generic."""
    hits: List[Tuple[str, str, int]] = []
    for key, phrases in table.items():
        best: Optional[str] = None
        for phrase in phrases:
            if phrase in text and (best is None or len(phrase) > len(best)):
                best = phrase
        if best:
            hits.append((key, best, len(best)))
    hits.sort(key=lambda h: -h[2])
    return [(k, p) for k, p, _ in hits]


def _segment_clauses(text: str) -> List[str]:
    """Split on connectives so a removal clause does not swallow the whole prompt."""
    return [s.strip() for s in re.split(r",| and | but | then | also |;", text) if s.strip()]


def _infer_site(text: str, classes: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    direct = _find_phrases(text, SITE_PHRASES)
    if direct:
        return direct[0][0], f'site "{direct[0][1]}" -> {direct[0][0]}'

    # Otherwise choose the site whose zones can actually host the requested hazards.
    if classes:
        scores: Dict[str, int] = {}
        for sid in SiteModel.ids():
            zones = {z.type for z in SiteModel.load(sid).zones}
            scores[sid] = sum(1 for c in classes
                              if set(HAZARD_BY_ID[c]["zone_types"]) & zones)
        best = max(scores, key=lambda s: scores[s])
        if scores[best]:
            return best, f"site inferred from requested hazards -> {best}"
    return None, None


def _parse_difficulty(text: str, spec: PromptSpec) -> None:
    for cue in HARDER_CUES:
        if cue in text:
            spec.difficulty_delta += 0.18
            spec.note(f'"{cue}" -> difficulty +0.18')
            return
    for cue in EASIER_CUES:
        if cue in text:
            spec.difficulty_delta -= 0.18
            spec.note(f'"{cue}" -> difficulty -0.18')
            return
    best: Optional[Tuple[float, str]] = None
    for value, phrases in DIFFICULTY_PHRASES:
        for phrase in phrases:
            if phrase in text and (best is None or len(phrase) > len(best[1])):
                best = (value, phrase)
    if best:
        spec.difficulty_target = best[0]
        spec.note(f'"{best[1]}" -> difficulty target {best[0]:.2f}')


def _parse_count(text: str, spec: PromptSpec) -> None:
    m = re.search(r"\b(\d+)\s+(?:hazards?|teaching points?|violations?|issues?|defects?)", text)
    if m:
        spec.n_hazards = max(1, min(6, int(m.group(1))))
        spec.note(f'"{m.group(0)}" -> {spec.n_hazards} teaching points')
        return
    for word, n in COUNT_WORDS.items():
        if re.search(rf"\b{word}\s+(?:hazards?|teaching points?|violations?|issues?|defects?)", text):
            spec.n_hazards = n
            spec.note(f'"{word}" -> {n} teaching points')
            return


def parse_lexicon(text: str, current: Optional[Scenario] = None) -> PromptSpec:
    raw = text
    text = _normalise(text)
    spec = PromptSpec(text=raw, backend="lexicon")

    # Intent: a modification needs both an existing scene and a modifying verb.
    if current is not None and any(v in text for v in MODIFY_VERBS):
        spec.intent = "modify"
        spec.note(f"modifying existing scenario {current.id}")

    # Split into clauses so "add a crane but remove the ladder" resolves correctly.
    for clause in _segment_clauses(text):
        removing = any(cue in clause for cue in REMOVE_CUES)
        for class_id, phrase in _find_phrases(clause, HAZARD_PHRASES):
            if removing:
                if class_id not in spec.remove_classes:
                    spec.remove_classes.append(class_id)
                    spec.note(f'"{phrase}" (negated) -> remove {class_id}')
            elif class_id not in spec.target_classes:
                spec.target_classes.append(class_id)
                spec.note(f'"{phrase}" -> {class_id}  [{HAZARD_BY_ID[class_id]["clause"]}]')

    _parse_difficulty(text, spec)
    _parse_count(text, spec)

    for trade, phrases in TRADE_PHRASES.items():
        for phrase in phrases:
            if phrase in text:
                spec.trade = trade
                spec.note(f'"{phrase}" -> trade {trade}')
                break
        if spec.trade:
            break

    if spec.intent == "create":
        site, why = _infer_site(text, spec.target_classes)
        if site:
            spec.site_template = site
            spec.note(why or "")
    else:
        direct = _find_phrases(text, SITE_PHRASES)
        if direct and current is not None and direct[0][0] != current.site_template:
            spec.site_template = direct[0][0]
            spec.note(f'site changed to {direct[0][0]} by "{direct[0][1]}"')

    if not spec.target_classes and not spec.remove_classes and spec.intent == "create":
        spec.note("no hazard class named, sampling from the classes this site can host")

    return spec


# --------------------------------------------------------------------------- #
# LLM backend
# --------------------------------------------------------------------------- #

PROMPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "target_classes"],
    "properties": {
        "intent": {"type": "string", "enum": ["create", "modify"]},
        "site_template": {"type": "string", "enum": SiteModel.ids()},
        "target_classes": {"type": "array", "items": {"type": "string", "enum": sorted(HAZARD_BY_ID)}},
        "remove_classes": {"type": "array", "items": {"type": "string", "enum": sorted(HAZARD_BY_ID)}},
        "difficulty_target": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "n_hazards": {"type": "integer", "minimum": 1, "maximum": 6},
    },
}


def parse_llm(text: str, current: Optional[Scenario] = None) -> PromptSpec:
    """Route the request through a schema-constrained model.

    The enums above mean the model cannot name a hazard class or a site that
    does not exist, so a hallucinated class fails at parse time rather than
    reaching the generator.
    """
    import json

    from .llm_backend import LLMBackend

    backend = LLMBackend()
    context = ""
    if current is not None:
        context = (f"\nThe scenario currently on screen is {current.id} on site "
                   f"{current.site_template} with teaching points: "
                   f"{', '.join(h.hazard_class for h in current.hazards)}.")
    messages = [
        {"role": "system",
         "content": "Translate a construction trainer's request into a scenario specification. "
                    "Reply with one JSON object against the schema. Use only enum values. "
                    "No prose."},
        {"role": "user", "content": f"Request: {text}{context}\n\nSchema: {json.dumps(PROMPT_SCHEMA)}"},
    ]
    data = json.loads(backend.complete(messages))

    spec = PromptSpec(text=text, backend="llm", intent=data.get("intent", "create"))
    spec.site_template = data.get("site_template")
    spec.target_classes = [c for c in data.get("target_classes", []) if c in HAZARD_BY_ID]
    spec.remove_classes = [c for c in data.get("remove_classes", []) if c in HAZARD_BY_ID]
    spec.difficulty_target = data.get("difficulty_target")
    spec.n_hazards = data.get("n_hazards")
    spec.note("parsed by schema-constrained model, values validated against the closed vocabulary")
    return spec


def parse(text: str, current: Optional[Scenario] = None, backend: str = "lexicon") -> PromptSpec:
    if backend == "llm":
        try:
            return parse_llm(text, current)
        except Exception as exc:  # falls back rather than failing the demonstration
            spec = parse_lexicon(text, current)
            spec.note(f"model backend unavailable ({exc.__class__.__name__}), used the lexicon")
            return spec
    return parse_lexicon(text, current)


# --------------------------------------------------------------------------- #
# Applying a spec
# --------------------------------------------------------------------------- #

def apply_to_scenario(spec: PromptSpec, current: Scenario) -> Dict[str, object]:
    """Fold a modification onto the scene already on screen.

    Returns generator keyword arguments. The modified request goes back through
    the full pipeline, so an edited scene is verified exactly as strictly as a
    new one: there is no path that reaches a trainee without passing the gate.
    """
    classes = [h.hazard_class for h in current.hazards]
    for c in spec.remove_classes:
        if c in classes:
            classes.remove(c)
    for c in spec.target_classes:
        if c not in classes:
            classes.append(c)

    site = spec.site_template or current.site_template
    zones = {z.type for z in SiteModel.load(site).zones}
    kept = [c for c in classes if set(HAZARD_BY_ID[c]["zone_types"]) & zones]
    dropped = [c for c in classes if c not in kept]
    if dropped:
        spec.note(f"{', '.join(dropped)} dropped: {site} has no zone that affords them")

    difficulty = current.difficulty_target
    if spec.difficulty_target is not None:
        difficulty = spec.difficulty_target
    difficulty = max(0.0, min(1.0, difficulty + spec.difficulty_delta))

    if spec.n_hazards is not None and len(kept) > spec.n_hazards:
        kept = kept[: spec.n_hazards]

    out: Dict[str, object] = {
        "site_template": site,
        "target_classes": kept[:6],
        "difficulty_target": round(difficulty, 3),
    }
    out["trade"] = spec.trade or current.trade
    if spec.n_hazards:
        out["n_hazards"] = spec.n_hazards
    return out


def to_generator_kwargs(spec: PromptSpec) -> Dict[str, object]:
    """Generator keyword arguments for a fresh scenario."""
    kwargs: Dict[str, object] = {}
    if spec.site_template:
        kwargs["site_template"] = spec.site_template
    if spec.target_classes:
        kwargs["target_classes"] = spec.target_classes[:6]
    if spec.difficulty_target is not None:
        kwargs["difficulty_target"] = max(0.0, min(1.0, spec.difficulty_target + spec.difficulty_delta))
    elif spec.difficulty_delta:
        kwargs["difficulty_target"] = max(0.0, min(1.0, 0.5 + spec.difficulty_delta))
    if spec.n_hazards:
        kwargs["n_hazards"] = spec.n_hazards
    if spec.trade:
        kwargs["trade"] = spec.trade
    return kwargs
