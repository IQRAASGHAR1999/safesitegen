"""Ablation study: what does each stage of the pipeline actually buy?

Four arms over the same scenario specifications and the same random seeds:

  A  unconstrained   no placement solver, no validation gate
  B  solver only     constrained placement, no validation gate
  C  gate only       unconstrained placement, gate plus repair
  D  full            constrained placement, gate plus repair

Reported per arm: first-pass validity, accepted rate after repair, which checks
fired, and the expressive range of the accepted set. Synthetic specification
noise (``--fault-rate``) stands in for generative hallucination so the gate has
something to catch; the grammar backend itself does not hallucinate.

    python experiments/run_ablation.py --n 400 --fault-rate 0.25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safesitegen.generator import generate                      # noqa: E402
from safesitegen.knowledge import HAZARD_BY_ID                  # noqa: E402
from safesitegen.metrics import summarise, svg_expressive_range  # noqa: E402
from safesitegen.site import SiteModel                          # noqa: E402

ARMS = [
    ("A_unconstrained", False, False),
    ("B_solver_only", True, False),
    ("C_gate_only", False, True),
    ("D_full", True, True),
]


def specs(args):
    """Sweep the specification space so expressive range is measurable."""
    rng = random.Random(args.seed)
    out = []
    for i in range(args.n):
        out.append({
            "site": SiteModel.ids()[i % len(SiteModel.ids())],
            "seed": args.seed + i,
            "hazards": rng.choice([2, 3, 4]),
            "difficulty": round(rng.uniform(0.2, 0.8), 3),
        })
    return out


def run_arm(name: str, use_solver: bool, use_gate: bool, args, plan) -> dict:
    scenarios, difficulties, first_pass, accepted = [], [], [], []
    failure_counter: Counter = Counter()
    attempts_total = 0

    for spec in plan:
        # First pass: identical specification, gate disabled, to read raw validity.
        raw = generate(site_template=spec["site"], seed=spec["seed"], n_hazards=spec["hazards"],
                       difficulty_target=spec["difficulty"], fault_rate=args.fault_rate,
                       use_solver=use_solver, use_gate=False)
        first_pass.append(raw.report.passed)
        for cid in set(raw.report.failure_ids()):
            failure_counter[cid] += 1

        result = generate(site_template=spec["site"], seed=spec["seed"], n_hazards=spec["hazards"],
                          difficulty_target=spec["difficulty"], fault_rate=args.fault_rate,
                          use_solver=use_solver, use_gate=use_gate)
        attempts_total += result.attempts
        # Without a gate nothing is screened out, so "accepted" is only meaningful
        # as "would survive the gate": score every arm against the same standard.
        ok = result.report.passed
        accepted.append(ok)
        if ok:
            scenarios.append(result.scenario)
            difficulties.append(result.report.difficulty)

    sites = {sid: SiteModel.load(sid) for sid in SiteModel.ids()}
    stats = summarise(scenarios, difficulties, [True] * len(scenarios),
                      sorted(HAZARD_BY_ID), sites=sites)
    n = len(plan)
    return {
        "arm": name,
        "use_solver": use_solver,
        "use_gate": use_gate,
        "n": n,
        "first_pass_validity": round(sum(first_pass) / n, 4),
        "accepted_rate": round(sum(accepted) / n, 4),
        "mean_attempts": round(attempts_total / n, 3),
        "top_failures": failure_counter.most_common(6),
        "accepted_stats": stats,
    }


def fault_detection(args, plan) -> dict:
    """Isolate the gate's catch rate on injected specification noise.

    Measured with the solver enabled so placement failures cannot be mistaken for
    hallucination catches, and counting only the regulatory layer, which is the
    layer a mislabelled teaching point should trip.
    """
    total = caught = 0
    by_kind: Counter = Counter()
    caught_by_kind: Counter = Counter()
    for spec in plan:
        r = generate(site_template=spec["site"], seed=spec["seed"], n_hazards=spec["hazards"],
                     difficulty_target=spec["difficulty"], fault_rate=args.fault_rate,
                     use_solver=True, use_gate=False)
        if not r.injected_faults:
            continue
        kinds = {f.split(":")[0] for f in r.injected_faults}
        reg_failed = any(c.failed for c in r.report.by_layer("regulatory"))
        total += 1
        for k in kinds:
            by_kind[k] += 1
            if reg_failed:
                caught_by_kind[k] += 1
        if reg_failed:
            caught += 1
    return {
        "scenarios_with_faults": total,
        "detection_rate": round(caught / total, 4) if total else None,
        "by_kind": {k: {"n": by_kind[k], "caught": caught_by_kind[k],
                        "rate": round(caught_by_kind[k] / by_kind[k], 4)} for k in by_kind},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--fault-rate", type=float, default=0.25)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    plan = specs(args)
    header = f"{'arm':<18}{'first-pass':>12}{'after repair':>14}{'attempts':>10}"
    print(header)
    print("-" * len(header))
    for name, solver, gate in ARMS:
        r = run_arm(name, solver, gate, args, plan)
        results.append(r)
        print(f"{name:<18}{r['first_pass_validity']:>11.1%}{r['accepted_rate']:>13.1%}"
              f"{r['mean_attempts']:>10.2f}")

    fd = fault_detection(args, plan)
    print()
    print("regulatory layer vs injected specification noise (solver on)")
    print(f"  scenarios carrying a fault  {fd['scenarios_with_faults']}")
    if fd["detection_rate"] is not None:
        print(f"  detected                    {fd['detection_rate']:.1%}")
    for kind, d in sorted(fd["by_kind"].items()):
        print(f"    {kind:<20}{d['caught']:>4} / {d['n']:<5}{d['rate']:.1%}")

    full = next(r for r in results if r["arm"] == "D_full")
    era = full["accepted_stats"]["expressive_range"]
    print()
    print("full pipeline, accepted set")
    print(f"  scenarios accepted        {full['accepted_stats']['n']}")
    print(f"  hazard class coverage     {full['accepted_stats']['class_coverage']:.1%}")
    print(f"  class entropy (normalised){full['accepted_stats']['class_entropy']:>8.3f}")
    print(f"  layout uniqueness         {full['accepted_stats']['signature_uniqueness']:.1%}")
    print(f"  expressive range coverage {era['coverage']:.1%} of the difficulty-dispersion grid")
    print(f"  largest single bin        {era['peak_share']:.1%} of samples")

    print()
    print("most frequent gate failures, unconstrained arm")
    for cid, count in results[0]["top_failures"]:
        print(f"  {cid:<32}{count:>5} / {args.n}")

    (out / "ablation.json").write_text(
        json.dumps({"arms": results, "fault_detection": fd}, indent=2), encoding="utf-8")
    (out / "expressive_range.svg").write_text(
        svg_expressive_range(era, "Full pipeline: accepted scenarios"), encoding="utf-8")
    print(f"\nwritten to {out / 'ablation.json'} and {out / 'expressive_range.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
