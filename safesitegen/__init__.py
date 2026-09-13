"""SafeSiteGen: verifiable procedural generation of construction safety training scenarios.

Reference implementation for the research proposal
"Verifiable Procedural Content Generation for Construction Safety Training".

Pipeline: specification -> hazard scenario graph -> constrained placement ->
three-layer validation gate -> runtime scene.
"""

from .schema import Entity, Hazard, Relation, Scenario, SchemaError
from .site import SiteModel
from .validate import ValidationReport, validate, evaluate_rule, audit, estimate_difficulty
from .layout import solve
from .generator import generate, GenerationResult
from .learner import LearnerModel, TrainingEvent
from .export import export_all, to_unity_scene

__version__ = "0.1.0"

__all__ = [
    "Entity",
    "Hazard",
    "Relation",
    "Scenario",
    "SchemaError",
    "SiteModel",
    "ValidationReport",
    "validate",
    "evaluate_rule",
    "audit",
    "estimate_difficulty",
    "solve",
    "generate",
    "GenerationResult",
    "LearnerModel",
    "TrainingEvent",
    "export_all",
    "to_unity_scene",
    "__version__",
]
