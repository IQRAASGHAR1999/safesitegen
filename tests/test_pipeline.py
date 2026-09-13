"""Validation gate, placement solver, learner model and end-to-end pipeline."""

import unittest

from safesitegen.generator import generate, inject_faults, build_scenario
from safesitegen.knowledge import HAZARD_BY_ID, RULES_BY_ID
from safesitegen.layout import solve, random_placement, energy
from safesitegen.learner import LearnerModel, TrainingEvent, P_INIT
from safesitegen.metrics import expressive_range, hazard_diversity, signature_uniqueness
from safesitegen.schema import Entity, Hazard, Scenario
from safesitegen.site import SiteModel
from safesitegen.validate import estimate_difficulty, validate


def minimal_scenario(site, controls=None, height=20.0):
    """One declared fall hazard on a slab edge, correctly realised."""
    s = Scenario(id="t1", site_template=site.id, trade="ironworker", activity="decking",
                 difficulty_target=0.5, target_hazard_classes=["fall_unprotected_edge"])
    s.entities = [
        Entity(id="w1", kind="worker", asset_type="ironworker",
               params={"context": "edge_exposure", "fall_height_ft": height,
                       "controls": controls or []}),
        Entity(id="d1", kind="structure", asset_type="material_pallet",
               params={}, is_distractor=True),
    ]
    s.hazards = [Hazard(id="h1", hazard_class="fall_unprotected_edge", target_entity="w1",
                        clause="1926.501(b)(1)", energy_source="gravitational",
                        required_controls=["guardrail_system", "safety_net",
                                           "personal_fall_arrest"],
                        cue_salience=0.55)]
    return s


class TestValidationGate(unittest.TestCase):
    def setUp(self):
        self.site = SiteModel.load("steel_frame_level3")

    def test_declared_hazard_that_is_not_realised_fails(self):
        s = minimal_scenario(self.site, controls=["guardrail_system"])
        solve(s, self.site, seed=1)
        report = validate(s, self.site)
        self.assertIn("REG.hazard_realised", report.failure_ids())

    def test_correctly_realised_hazard_passes_the_regulatory_layer(self):
        s = minimal_scenario(self.site)
        solve(s, self.site, seed=1)
        report = validate(s, self.site)
        self.assertNotIn("REG.hazard_realised", report.failure_ids())
        self.assertNotIn("REG.no_unintended_violation", report.failure_ids())

    def test_undeclared_violation_is_caught(self):
        s = minimal_scenario(self.site)
        s.entities.append(Entity(id="g1", kind="control", asset_type="guardrail_system",
                                 params={"top_rail_height_in": 28.0}))
        solve(s, self.site, seed=1)
        report = validate(s, self.site)
        self.assertIn("REG.no_unintended_violation", report.failure_ids())

    def test_compliant_distractor_of_a_regulated_type_does_not_trip_the_gate(self):
        s = minimal_scenario(self.site)
        s.entities.append(Entity(id="g1", kind="control", asset_type="guardrail_system",
                                 params={"top_rail_height_in": 42.0}, is_distractor=True))
        solve(s, self.site, seed=1)
        report = validate(s, self.site)
        self.assertNotIn("REG.no_unintended_violation", report.failure_ids())

    def test_out_of_bounds_placement_is_caught(self):
        s = minimal_scenario(self.site)
        for e in s.entities:
            e.x, e.y = 999.0, 999.0
        report = validate(s, self.site)
        self.assertIn("PHY.in_bounds", report.failure_ids())

    def test_overlapping_entities_are_caught(self):
        s = minimal_scenario(self.site)
        for e in s.entities:
            e.x, e.y = 5.0, 5.0
        report = validate(s, self.site)
        self.assertIn("PHY.no_overlap", report.failure_ids())

    def test_hazard_without_its_affordance_is_caught(self):
        s = minimal_scenario(self.site)
        solve(s, self.site, seed=1)
        # move the edge-fall hazard onto the laydown area, where no edge exists
        s.entity("w1").x, s.entity("w1").y = 12.0, 14.0
        report = validate(s, self.site)
        self.assertIn("PHY.zone_affordance", report.failure_ids())

    def test_missing_distractor_is_caught(self):
        s = minimal_scenario(self.site)
        s.entities = [e for e in s.entities if not e.is_distractor]
        solve(s, self.site, seed=1)
        report = validate(s, self.site)
        self.assertIn("PED.distractor_present", report.failure_ids())

    def test_unplaced_entity_fails(self):
        s = minimal_scenario(self.site)
        report = validate(s, self.site)
        self.assertIn("PHY.placed", report.failure_ids())

    def test_report_is_serialisable_and_traceable(self):
        s = minimal_scenario(self.site)
        solve(s, self.site, seed=1)
        d = validate(s, self.site).to_dict()
        self.assertIn("checks", d)
        self.assertTrue(any(c.get("clause") for c in d["checks"]))


class TestLayout(unittest.TestCase):
    def setUp(self):
        self.site = SiteModel.load("steel_frame_level3")

    def test_solver_is_deterministic_under_a_seed(self):
        a, _ = solve(minimal_scenario(self.site), self.site, seed=42)
        b, _ = solve(minimal_scenario(self.site), self.site, seed=42)
        self.assertEqual([(e.id, e.x, e.y) for e in a.entities],
                         [(e.id, e.x, e.y) for e in b.entities])

    def test_solver_beats_random_placement_on_energy(self):
        import random
        rng = random.Random(0)
        solved, e_solved = solve(minimal_scenario(self.site), self.site, seed=3)
        rand = random_placement(minimal_scenario(self.site), self.site, rng)
        allowed = {ent.id: set(self.site.all_free_cells()) for ent in rand.entities}
        self.assertLessEqual(e_solved, energy(rand, self.site, allowed))

    def test_solver_places_every_entity_in_bounds(self):
        s, _ = solve(minimal_scenario(self.site), self.site, seed=9)
        for e in s.entities:
            self.assertTrue(self.site.in_bounds(e.x, e.y), e.id)

    def test_difficulty_is_bounded(self):
        s, _ = solve(minimal_scenario(self.site), self.site, seed=5)
        d = estimate_difficulty(s, self.site)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)


class TestSite(unittest.TestCase):
    def test_all_templates_load_and_are_self_consistent(self):
        for sid in SiteModel.ids():
            site = SiteModel.load(sid)
            self.assertTrue(site.walkable(*site.entry), sid)
            self.assertTrue(site.patrol, sid)
            for p in site.patrol:
                self.assertTrue(site.in_bounds(*p), f"{sid} patrol {p}")

    def test_entry_reaches_every_patrol_point(self):
        for sid in SiteModel.ids():
            site = SiteModel.load(sid)
            reachable = site.reachable_cells()
            for p in site.patrol:
                self.assertIn(tuple(p), reachable, f"{sid} {p}")

    def test_obstacles_occlude_sightlines(self):
        site = SiteModel.load("steel_frame_level3")
        self.assertFalse(site.has_line_of_sight((7, 6), (9, 6)))
        self.assertTrue(site.has_line_of_sight((2, 3), (4, 3)))


class TestLearner(unittest.TestCase):
    def test_a_miss_lowers_mastery_relative_to_a_hit(self):
        hit = LearnerModel("A")
        miss = LearnerModel("B")
        hit.observe(TrainingEvent("fall_unprotected_edge", True))
        miss.observe(TrainingEvent("fall_unprotected_edge", False))
        self.assertGreater(hit.mastery["fall_unprotected_edge"],
                           miss.mastery["fall_unprotected_edge"])

    def test_repeated_hits_increase_mastery_monotonically(self):
        m = LearnerModel("A")
        last = m.mastery["rebar_impalement"]
        for _ in range(6):
            m.observe(TrainingEvent("rebar_impalement", True))
            self.assertGreater(m.mastery["rebar_impalement"], last)
            last = m.mastery["rebar_impalement"]

    def test_mastery_stays_a_probability(self):
        m = LearnerModel("A")
        for i in range(40):
            m.observe(TrainingEvent("fall_floor_opening", i % 3 == 0))
            self.assertTrue(0.0 <= m.mastery["fall_floor_opening"] <= 1.0)

    def test_weakest_classes_are_targeted_first(self):
        m = LearnerModel("A")
        for _ in range(10):
            m.observe(TrainingEvent("fall_unprotected_edge", True))
        weakest = m.weakest_classes(k=3)
        self.assertNotIn("fall_unprotected_edge", weakest)

    def test_recent_misses_get_priority(self):
        m = LearnerModel("A")
        for cls in HAZARD_BY_ID:
            m.observe(TrainingEvent(cls, True))
        m.observe(TrainingEvent("rebar_impalement", False))
        self.assertIn("rebar_impalement", m.weakest_classes(k=3))

    def test_difficulty_targeting_moves_with_mastery(self):
        novice = LearnerModel("N")
        expert = LearnerModel("E")
        for _ in range(15):
            expert.observe(TrainingEvent("fall_unprotected_edge", True))
        self.assertGreater(expert.difficulty_for(["fall_unprotected_edge"]),
                           novice.difficulty_for(["fall_unprotected_edge"]))

    def test_seen_layouts_are_not_novel(self):
        m = LearnerModel("A")
        m.observe(TrainingEvent("fall_unprotected_edge", True, scenario_signature="abc123"))
        self.assertFalse(m.is_novel("abc123"))
        self.assertTrue(m.is_novel("def456"))

    def test_learner_roundtrip(self):
        m = LearnerModel("A")
        m.observe(TrainingEvent("fall_unprotected_edge", False, scenario_signature="s1"))
        clone = LearnerModel.from_dict(m.to_dict())
        self.assertEqual(clone.trainee_id, "A")
        self.assertAlmostEqual(clone.mastery["fall_unprotected_edge"],
                               m.mastery["fall_unprotected_edge"])
        self.assertEqual(clone.seen_signatures, ["s1"])

    def test_next_spec_shape(self):
        spec = LearnerModel("A").next_spec(k=3)
        self.assertEqual(len(spec["target_classes"]), 3)
        self.assertTrue(0.0 <= float(spec["difficulty_target"]) <= 1.0)


class TestFaultInjection(unittest.TestCase):
    def setUp(self):
        self.site = SiteModel.load("steel_frame_level3")

    def test_zero_rate_injects_nothing(self):
        import random
        s = minimal_scenario(self.site)
        self.assertEqual(inject_faults(s, 0.0, random.Random(0)), [])

    def test_phantom_hazard_is_detected_by_the_gate(self):
        import random
        s = minimal_scenario(self.site)
        rule = RULES_BY_ID["1926.501(b)(1)"]
        control = rule["predicate"]["controls_any"][0]
        s.entity("w1").params["controls"] = [control]
        s.hazards[0].present_controls = [control]
        solve(s, self.site, seed=2)
        self.assertIn("REG.hazard_realised", validate(s, self.site).failure_ids())


class TestPipeline(unittest.TestCase):
    def test_clean_generation_is_accepted(self):
        r = generate(site_template="steel_frame_level3", seed=5, fault_rate=0.0)
        self.assertTrue(r.report.passed, r.report.summary())

    def test_generation_is_reproducible(self):
        a = generate(site_template="utility_trench_corridor", seed=17, fault_rate=0.2)
        b = generate(site_template="utility_trench_corridor", seed=17, fault_rate=0.2)
        self.assertEqual(a.scenario.configuration_signature(),
                         b.scenario.configuration_signature())

    def test_every_site_template_generates(self):
        for sid in SiteModel.ids():
            r = generate(site_template=sid, seed=11, fault_rate=0.0)
            self.assertTrue(r.report.passed, f"{sid}: {r.report.summary()}")

    def test_explicit_class_request_is_delivered(self):
        r = generate(site_template="utility_trench_corridor",
                     target_classes=["trench_no_protective_system", "trench_spoil_too_close"],
                     seed=4)
        delivered = {h.hazard_class for h in r.scenario.hazards}
        self.assertEqual(delivered, {"trench_no_protective_system", "trench_spoil_too_close"})

    def test_gate_raises_validity_over_the_unconstrained_baseline(self):
        def rate(**kw):
            return sum(generate(seed=1000 + i, fault_rate=0.25, **kw).report.passed
                       for i in range(40)) / 40
        baseline = rate(use_solver=False, use_gate=False)
        full = rate(use_solver=True, use_gate=True)
        self.assertGreater(full, baseline + 0.4)

    def test_repair_records_what_it_changed(self):
        found = False
        for seed in range(60):
            r = generate(seed=seed, fault_rate=0.6)
            if r.repairs_applied:
                found = True
                self.assertTrue(all(":" in x or x == "add_distractor" for x in r.repairs_applied))
                break
        self.assertTrue(found, "no repair was exercised in 60 samples")

    def test_export_writes_all_artefacts(self):
        import json
        import tempfile
        from pathlib import Path
        from safesitegen.export import export_all
        r = generate(site_template="steel_frame_level3", seed=8)
        with tempfile.TemporaryDirectory() as td:
            paths = export_all(r.scenario, r.report, td)
            for key, p in paths.items():
                self.assertTrue(Path(p).exists(), key)
            scene = json.loads(Path(paths["scene"]).read_text())
            self.assertEqual(len(scene["answerKey"]), len(r.scenario.hazards))
            self.assertIn("<canvas", Path(paths["viewer"]).read_text())


class TestMetrics(unittest.TestCase):
    def test_expressive_range_counts_every_sample(self):
        era = expressive_range([(0.1, 0.2), (0.9, 0.9), (0.5, 0.5)], bins=10)
        self.assertEqual(era["n"], 3)
        self.assertAlmostEqual(era["coverage"], 3 / 100)

    def test_expressive_range_clamps_out_of_range_points(self):
        era = expressive_range([(1.5, -0.4)], bins=5)
        self.assertEqual(era["n"], 1)

    def test_repeated_classes_lower_diversity(self):
        s = Scenario(id="x", site_template="steel_frame_level3", trade="a", activity="b")
        s.entities = [Entity(id="e1", kind="worker", asset_type="w"),
                      Entity(id="e2", kind="worker", asset_type="w")]
        s.hazards = [
            Hazard(id="h1", hazard_class="fall_unprotected_edge", target_entity="e1",
                   clause="1926.501(b)(1)", energy_source="gravitational"),
            Hazard(id="h2", hazard_class="fall_unprotected_edge", target_entity="e2",
                   clause="1926.501(b)(1)", energy_source="gravitational"),
        ]
        self.assertAlmostEqual(hazard_diversity(s), 0.5)

    def test_distinct_layouts_are_unique(self):
        scenarios = [generate(seed=s, fault_rate=0.0).scenario for s in range(6)]
        self.assertGreater(signature_uniqueness(scenarios), 0.9)


class TestKnowledgeIntegrity(unittest.TestCase):
    def test_every_hazard_class_cites_a_rule_in_the_pack(self):
        for cid, meta in HAZARD_BY_ID.items():
            self.assertIn(meta["clause"], RULES_BY_ID, cid)

    def test_every_class_has_a_victim_entity_template(self):
        for cid, meta in HAZARD_BY_ID.items():
            roles = [e["role"] for e in meta["entities"]]
            self.assertIn("victim", roles, cid)

    def test_required_controls_match_the_rule_predicate(self):
        for cid, meta in HAZARD_BY_ID.items():
            pred = RULES_BY_ID[meta["clause"]]["predicate"]
            expected = pred.get("controls_any", [])
            if expected:
                self.assertEqual(set(meta["required_controls"]), set(expected), cid)

    def test_every_class_names_a_zone_that_some_site_provides(self):
        provided = {z.type for sid in SiteModel.ids() for z in SiteModel.load(sid).zones}
        for cid, meta in HAZARD_BY_ID.items():
            self.assertTrue(set(meta["zone_types"]) & provided, cid)


if __name__ == "__main__":
    unittest.main()
