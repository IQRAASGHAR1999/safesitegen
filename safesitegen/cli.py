"""Command line interface: ``python -m safesitegen ...``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .export import export_all
from .generator import generate
from .prompt import apply_to_scenario, parse, to_generator_kwargs
from .schema import Scenario
from .knowledge import HAZARD_BY_ID, RULE_META, RULES
from .learner import LearnerModel, TrainingEvent
from .site import SiteModel


def cmd_generate(args: argparse.Namespace) -> int:
    result = generate(
        site_template=args.site,
        target_classes=args.classes or None,
        difficulty_target=args.difficulty,
        seed=args.seed,
        n_hazards=args.hazards,
        n_distractors=args.distractors,
        fault_rate=args.fault_rate,
        use_solver=not args.no_solver,
        use_gate=not args.no_gate,
    )
    site = SiteModel.load(result.scenario.site_template)

    print(f"scenario   {result.scenario.id}")
    print(f"site       {site.name}  ({site.width} x {site.depth} m)")
    print(f"task       {result.scenario.trade.replace('_', ' ')}, {result.scenario.activity}")
    print(f"teaching   {', '.join(h.hazard_class for h in result.scenario.hazards)}")
    print(f"attempts   {result.attempts}   energy {result.energy:.2f}")
    if result.injected_faults:
        print(f"injected   {', '.join(result.injected_faults)}")
    if result.repairs_applied:
        print(f"repairs    {', '.join(result.repairs_applied)}")
    print()
    print(result.report.summary())

    if args.out:
        paths = export_all(result.scenario, result.report, args.out, site)
        print()
        for key, path in paths.items():
            print(f"{key:9} {path}")
    return 0 if result.accepted else 1


def cmd_prompt(args: argparse.Namespace) -> int:
    """Natural language in, validated training scene out."""
    current = None
    if args.modify:
        current = Scenario.from_dict(json.loads(Path(args.modify).read_text()))

    spec = parse(args.text, current, backend=args.backend)
    print(spec.explain())
    print()

    if spec.intent == "modify" and current is not None:
        kwargs = apply_to_scenario(spec, current)
        print("resolved   " + json.dumps(kwargs))
    else:
        kwargs = to_generator_kwargs(spec)
        print("resolved   " + json.dumps(kwargs))
    print()

    result = generate(seed=args.seed, fault_rate=args.fault_rate,
                      use_gate=not args.no_gate, **kwargs)
    site = SiteModel.load(result.scenario.site_template)

    print(f"scenario   {result.scenario.id}")
    print(f"site       {site.name}")
    print(f"task       {result.scenario.trade.replace('_', ' ')}, {result.scenario.activity}")
    for h in result.scenario.hazards:
        e = result.scenario.entity(h.target_entity)
        print(f"  hazard   {h.hazard_class:<32} {h.clause:<18} at ({int(e.x)}, {int(e.y)})")
    prov = result.scenario.provenance
    if prov.get("difficulty_clamped_to") is not None:
        lo, hi = prov["difficulty_feasible_range"]
        print(f"note       requested difficulty {prov['difficulty_requested']:.2f} is outside what "
              f"this site and hazard set can reach ([{lo:.2f}, {hi:.2f}]); "
              f"clamped to {prov['difficulty_clamped_to']:.2f}")
    if result.injected_faults:
        print(f"injected   {', '.join(result.injected_faults)}")
    if result.repairs_applied:
        print(f"repairs    {', '.join(result.repairs_applied)}")
    print(f"attempts   {result.attempts}")
    print()
    print(result.report.summary())

    if args.out:
        paths = export_all(result.scenario, result.report, args.out, site)
        print()
        for key, path in paths.items():
            print(f"{key:9} {path}")
    return 0 if result.accepted else 1


def cmd_rules(args: argparse.Namespace) -> int:
    print(RULE_META["disclaimer"])
    print()
    for rule in RULES:
        print(f"{rule['id']:<20} [{rule['subpart']}] {rule['topic']}")
        if args.verbose:
            print(f"{'':<20} {rule['summary']}")
            print(f"{'':<20} predicate: {json.dumps(rule['predicate'])}")
    return 0


def cmd_classes(_: argparse.Namespace) -> int:
    for class_id, meta in HAZARD_BY_ID.items():
        print(f"{class_id:<34} {meta['clause']:<18} {meta['name']}")
    return 0


def cmd_sites(_: argparse.Namespace) -> int:
    for site_id in SiteModel.ids():
        site = SiteModel.load(site_id)
        print(f"{site_id:<26} {site.width:>3} x {site.depth:<3} m   {site.name}")
    return 0


def cmd_adapt(args: argparse.Namespace) -> int:
    """Simulate a trainee across sessions and show the curriculum adapting."""
    model = LearnerModel(trainee_id=args.trainee)
    import random

    rng = random.Random(args.seed)
    print(f"{'session':<8}{'targeted classes':<62}{'diff':<7}{'detected'}")
    print("-" * 92)
    for session in range(1, args.sessions + 1):
        spec = model.next_spec(k=args.hazards)
        classes = spec["target_classes"]  # type: ignore[index]
        difficulty = float(spec["difficulty_target"])  # type: ignore[arg-type]
        result = generate(
            target_classes=classes,
            difficulty_target=difficulty,
            seed=args.seed + session * 31,
            n_hazards=args.hazards,
            fault_rate=args.fault_rate,
        )
        if not model.is_novel(result.scenario.configuration_signature()):
            continue
        outcomes = []
        for cls in classes:
            p = model.predict_detection(cls, result.report.difficulty)
            detected = rng.random() < p
            outcomes.append("hit" if detected else "MISS")
            model.observe(
                TrainingEvent(
                    hazard_class=cls,
                    detected=detected,
                    scenario_signature=result.scenario.configuration_signature(),
                ),
                difficulty=result.report.difficulty,
            )
        shown = ", ".join(c.replace("_", " ") for c in classes)
        print(f"{session:<8}{shown[:60]:<62}{difficulty:<7.2f}{' '.join(outcomes)}")

    print()
    print("posterior mastery by class")
    for cls, m in sorted(model.mastery.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(m * 34)
        print(f"  {cls:<34}{m:5.2f}  {bar}")

    if args.out:
        Path(args.out).write_text(model.to_json(), encoding="utf-8")
        print(f"\nlearner model written to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="safesitegen", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate, validate and export one scenario")
    g.add_argument("--site", default=None, help="site template id (default: random)")
    g.add_argument("--classes", nargs="*", default=None, help="explicit hazard classes")
    g.add_argument("--hazards", type=int, default=3)
    g.add_argument("--distractors", type=int, default=3)
    g.add_argument("--difficulty", type=float, default=0.5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--fault-rate", type=float, default=0.0,
                   help="synthetic specification-noise rate (hallucination model)")
    g.add_argument("--no-solver", action="store_true", help="ablate the placement solver")
    g.add_argument("--no-gate", action="store_true", help="ablate the validation gate")
    g.add_argument("--out", default=None, help="output directory")
    g.set_defaults(func=cmd_generate)

    pr = sub.add_parser("prompt", help="generate or modify a scene from natural language")
    pr.add_argument("text", help="what you want, in plain English")
    pr.add_argument("--modify", default=None,
                    help="path to an existing scenario.json to edit instead of starting fresh")
    pr.add_argument("--backend", choices=["lexicon", "llm"], default="lexicon")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--fault-rate", type=float, default=0.0)
    pr.add_argument("--no-gate", action="store_true",
                    help="ablate the validation gate, to show what would ship without it")
    pr.add_argument("--out", default=None, help="output directory")
    pr.set_defaults(func=cmd_prompt)

    r = sub.add_parser("rules", help="list the machine-checkable rule pack")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_rules)

    c = sub.add_parser("classes", help="list hazard classes")
    c.set_defaults(func=cmd_classes)

    s = sub.add_parser("sites", help="list site substrates")
    s.set_defaults(func=cmd_sites)

    a = sub.add_parser("adapt", help="simulate adaptive sessions for one trainee")
    a.add_argument("--trainee", default="T001")
    a.add_argument("--sessions", type=int, default=12)
    a.add_argument("--hazards", type=int, default=3)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--fault-rate", type=float, default=0.0)
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_adapt)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
