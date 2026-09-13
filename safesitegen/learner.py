"""Learner model: what to generate next, for this particular trainee.

Hazard recognition is not one skill. A worker who reliably spots an unprotected
edge may never notice a spoil pile surcharging a trench, so mastery is tracked
per hazard class using Bayesian knowledge tracing, with the hazard taxonomy as
the skill set.

Selection then does three things at once:

  * target the classes with the weakest posterior mastery,
  * set a difficulty that puts predicted detection near the desirable-difficulty
    band rather than at either ceiling, and
  * refuse layouts the trainee has already seen, so what is learned is the
    hazard and not the room.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .knowledge import HAZARD_BY_ID

P_INIT = 0.25      # prior mastery before any evidence
P_TRANSIT = 0.20   # chance of learning a class from one exposure
P_SLIP = 0.08      # knows it, still misses it, on the easiest scene
P_GUESS = 0.15     # does not know it, flags it anyway
SLIP_SLOPE = 0.45  # how much harder scenes erode a mastered response
GUESS_SLOPE = 0.60 # how much harder scenes suppress lucky guesses

TARGET_DETECTION = 0.70


@dataclass
class TrainingEvent:
    hazard_class: str
    detected: bool
    latency_s: float = 0.0
    scenario_signature: str = ""


@dataclass
class LearnerModel:
    trainee_id: str
    mastery: Dict[str, float] = field(default_factory=dict)
    exposures: Dict[str, int] = field(default_factory=dict)
    seen_signatures: List[str] = field(default_factory=list)
    history: List[TrainingEvent] = field(default_factory=list)

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        for class_id in HAZARD_BY_ID:
            self.mastery.setdefault(class_id, P_INIT)
            self.exposures.setdefault(class_id, 0)

    # ------------------------------------------------------------- inference
    def predict_detection(self, hazard_class: str, difficulty: float = 0.5) -> float:
        """P(trainee flags this class) given current mastery and scene difficulty."""
        L = self.mastery.get(hazard_class, P_INIT)
        slip = min(0.95, P_SLIP + SLIP_SLOPE * difficulty)
        guess = max(0.02, P_GUESS * (1.0 - GUESS_SLOPE * difficulty))
        return L * (1.0 - slip) + (1.0 - L) * guess

    def observe(self, event: TrainingEvent, difficulty: float = 0.5) -> float:
        """Bayesian knowledge tracing update. Returns the posterior mastery."""
        cls = event.hazard_class
        L = self.mastery.get(cls, P_INIT)
        slip = min(0.95, P_SLIP + SLIP_SLOPE * difficulty)
        guess = max(0.02, P_GUESS * (1.0 - GUESS_SLOPE * difficulty))

        if event.detected:
            num = L * (1.0 - slip)
            den = num + (1.0 - L) * guess
        else:
            num = L * slip
            den = num + (1.0 - L) * (1.0 - guess)

        posterior = num / den if den > 0 else L
        self.mastery[cls] = posterior + (1.0 - posterior) * P_TRANSIT
        self.exposures[cls] = self.exposures.get(cls, 0) + 1
        self.history.append(event)
        if event.scenario_signature and event.scenario_signature not in self.seen_signatures:
            self.seen_signatures.append(event.scenario_signature)
        return self.mastery[cls]

    # ------------------------------------------------------------- selection
    def weakest_classes(self, k: int = 3, pool: Optional[Sequence[str]] = None) -> List[str]:
        candidates = list(pool) if pool else list(HAZARD_BY_ID)
        # spaced repetition: recently missed classes get priority over merely unseen ones
        recent_misses = {e.hazard_class for e in self.history[-8:] if not e.detected}
        def key(c: str) -> Tuple[float, int]:
            penalty = -0.25 if c in recent_misses else 0.0
            return (self.mastery.get(c, P_INIT) + penalty, self.exposures.get(c, 0))
        return sorted(candidates, key=key)[:k]

    def difficulty_for(self, classes: Sequence[str]) -> float:
        """Pick the difficulty that puts predicted detection nearest the target band."""
        best_d, best_gap = 0.5, 1e9
        for step in range(21):
            d = step / 20.0
            preds = [self.predict_detection(c, d) for c in classes]
            mean_pred = sum(preds) / len(preds) if preds else 0.5
            gap = abs(mean_pred - TARGET_DETECTION)
            if gap < best_gap:
                best_gap, best_d = gap, d
        return best_d

    def is_novel(self, signature: str) -> bool:
        return signature not in self.seen_signatures

    def next_spec(self, k: int = 3, pool: Optional[Sequence[str]] = None) -> Dict[str, object]:
        classes = self.weakest_classes(k, pool)
        return {
            "target_classes": classes,
            "difficulty_target": round(self.difficulty_for(classes), 3),
            "rationale": {
                c: {
                    "mastery": round(self.mastery.get(c, P_INIT), 3),
                    "exposures": self.exposures.get(c, 0),
                }
                for c in classes
            },
        }

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["history"] = [asdict(e) for e in self.history]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LearnerModel":
        model = cls(trainee_id=str(data["trainee_id"]))
        model.mastery.update(data.get("mastery", {}))  # type: ignore[arg-type]
        model.exposures.update(data.get("exposures", {}))  # type: ignore[arg-type]
        model.seen_signatures = list(data.get("seen_signatures", []))  # type: ignore[arg-type]
        model.history = [TrainingEvent(**e) for e in data.get("history", [])]  # type: ignore[arg-type]
        return model
