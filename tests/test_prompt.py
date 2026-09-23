"""Natural language parsing and scene modification."""

import json
import tempfile
import unittest
from pathlib import Path

from safesitegen.generator import generate
from safesitegen.knowledge import HAZARD_BY_ID
from safesitegen.prompt import (
    apply_to_scenario, parse, parse_lexicon, to_generator_kwargs,
)
from safesitegen.schema import Scenario
from safesitegen.site import SiteModel


class TestHazardRecognition(unittest.TestCase):
    def test_named_hazards_are_recognised(self):
        spec = parse_lexicon("unshored trench with the spoil piled on the edge")
        self.assertIn("trench_no_protective_system", spec.target_classes)
        self.assertIn("trench_spoil_too_close", spec.target_classes)

    def test_every_class_is_reachable_by_at_least_one_phrase(self):
        from safesitegen.prompt import HAZARD_PHRASES
        for class_id in HAZARD_BY_ID:
            self.assertIn(class_id, HAZARD_PHRASES, f"{class_id} has no phrases")
            phrase = HAZARD_PHRASES[class_id][0]
            spec = parse_lexicon(f"a site with {phrase}")
            self.assertIn(class_id, spec.target_classes, f"{class_id} via '{phrase}'")

    def test_parser_cannot_invent_a_class(self):
        spec = parse_lexicon("a dragon attacking the site with radioactive scaffolding")
        for c in spec.target_classes:
            self.assertIn(c, HAZARD_BY_ID)

    def test_trace_is_populated(self):
        spec = parse_lexicon("unshored trench, make it hard")
        self.assertTrue(spec.trace)
        self.assertIn("prompt", spec.explain())


class TestModifiers(unittest.TestCase):
    def test_difficulty_words(self):
        self.assertLess(parse_lexicon("keep it easy").difficulty_target, 0.4)
        self.assertGreater(parse_lexicon("make it hard for an experienced crew").difficulty_target, 0.6)

    def test_relative_difficulty(self):
        self.assertGreater(parse_lexicon("make it harder").difficulty_delta, 0)
        self.assertLess(parse_lexicon("make it easier").difficulty_delta, 0)

    def test_hazard_count(self):
        self.assertEqual(parse_lexicon("give me 4 hazards").n_hazards, 4)
        self.assertEqual(parse_lexicon("three hazards please").n_hazards, 3)

    def test_count_is_clamped(self):
        self.assertLessEqual(parse_lexicon("show me 99 hazards").n_hazards, 6)

    def test_trade_detection(self):
        self.assertEqual(parse_lexicon("a storm drain crew at work").trade, "pipelayer")
        self.assertEqual(parse_lexicon("ironworker connecting steel").trade, "ironworker")


class TestSiteInference(unittest.TestCase):
    def test_explicit_site_words(self):
        self.assertEqual(parse_lexicon("a highway lane closure").site_template,
                         "highway_workzone_taper")
        self.assertEqual(parse_lexicon("a trench job").site_template,
                         "utility_trench_corridor")

    def test_site_inferred_from_hazards_when_unnamed(self):
        spec = parse_lexicon("someone working with no cave-in protection")
        self.assertIsNotNone(spec.site_template)
        zones = {z.type for z in SiteModel.load(spec.site_template).zones}
        self.assertTrue(set(HAZARD_BY_ID["trench_no_protective_system"]["zone_types"]) & zones)

    def test_inferred_site_can_host_every_requested_class(self):
        spec = parse_lexicon("scaffold with no guardrail")
        zones = {z.type for z in SiteModel.load(spec.site_template).zones}
        self.assertIn("scaffold_bay", zones)


class TestIntent(unittest.TestCase):
    def setUp(self):
        self.current = generate(site_template="utility_trench_corridor",
                                target_classes=["trench_no_protective_system",
                                                "trench_spoil_too_close"],
                                seed=3).scenario

    def test_create_without_an_existing_scene(self):
        self.assertEqual(parse_lexicon("add a ladder hazard").intent, "create")

    def test_modify_with_an_existing_scene(self):
        self.assertEqual(parse_lexicon("now add a ladder hazard", self.current).intent, "modify")

    def test_removal_is_separated_from_addition(self):
        spec = parse_lexicon("add a crane near the power line but remove the spoil pile",
                             self.current)
        self.assertIn("power_line_encroachment", spec.target_classes)
        self.assertIn("trench_spoil_too_close", spec.remove_classes)
        self.assertNotIn("trench_spoil_too_close", spec.target_classes)

    def test_modification_folds_onto_the_existing_scene(self):
        spec = parse_lexicon("now add a crane near the power line, remove the spoil pile",
                             self.current)
        kwargs = apply_to_scenario(spec, self.current)
        self.assertIn("power_line_encroachment", kwargs["target_classes"])
        self.assertIn("trench_no_protective_system", kwargs["target_classes"])
        self.assertNotIn("trench_spoil_too_close", kwargs["target_classes"])

    def test_modification_keeps_the_trade(self):
        spec = parse_lexicon("make it harder", self.current)
        self.assertEqual(apply_to_scenario(spec, self.current)["trade"], self.current.trade)

    def test_modification_drops_classes_the_site_cannot_host(self):
        spec = parse_lexicon("add a scaffold with no guardrail", self.current)
        kwargs = apply_to_scenario(spec, self.current)
        self.assertNotIn("scaffold_missing_guardrail", kwargs["target_classes"])


class TestEndToEnd(unittest.TestCase):
    def test_prompt_produces_a_validated_scene(self):
        spec = parse_lexicon("ironworker on a steel deck with an unprotected leading edge "
                             "and an uncovered floor opening")
        result = generate(seed=5, **to_generator_kwargs(spec))
        self.assertTrue(result.report.passed, result.report.summary())
        delivered = {h.hazard_class for h in result.scenario.hazards}
        self.assertIn("fall_unprotected_edge", delivered)
        self.assertIn("fall_floor_opening", delivered)

    def test_modified_scene_is_validated_as_strictly(self):
        base = generate(site_template="steel_frame_level3",
                        target_classes=["fall_unprotected_edge"], seed=2).scenario
        spec = parse_lexicon("now also add an uncovered floor opening", base)
        result = generate(seed=2, **apply_to_scenario(spec, base))
        self.assertTrue(result.report.passed, result.report.summary())

    def test_impossible_difficulty_is_clamped_not_failed(self):
        result = generate(site_template="utility_trench_corridor",
                          target_classes=["trench_spoil_too_close"],
                          difficulty_target=0.99, seed=1)
        self.assertTrue(result.report.passed, result.report.summary())
        self.assertIsNotNone(result.scenario.provenance["difficulty_clamped_to"])

    def test_feasible_range_is_reported(self):
        result = generate(site_template="steel_frame_level3", seed=1)
        lo, hi = result.scenario.provenance["difficulty_feasible_range"]
        self.assertLessEqual(lo, hi)
        self.assertTrue(0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0)

    def test_named_classes_without_a_count_are_delivered_exactly(self):
        result = generate(site_template="steel_frame_level3",
                          target_classes=["fall_unprotected_edge", "rebar_impalement"], seed=6)
        self.assertEqual(len(result.scenario.hazards), 2)

    def test_explicit_count_tops_up_named_classes(self):
        result = generate(site_template="steel_frame_level3",
                          target_classes=["fall_unprotected_edge"], n_hazards=3, seed=6)
        self.assertEqual(len(result.scenario.hazards), 3)
        self.assertIn("fall_unprotected_edge",
                      [h.hazard_class for h in result.scenario.hazards])

    def test_llm_backend_falls_back_rather_than_crashing(self):
        spec = parse("a trench with no shoring", backend="llm")
        self.assertIn("trench_no_protective_system", spec.target_classes)


class TestEnvironmentExport(unittest.TestCase):
    def test_environment_is_written_and_self_contained(self):
        result = generate(site_template="steel_frame_level3", seed=4)
        from safesitegen.export import export_all
        with tempfile.TemporaryDirectory() as td:
            paths = export_all(result.scenario, result.report, td)
            html = Path(paths["environment"]).read_text()
            self.assertIn("<canvas", html)
            self.assertIn("answerKey", html)
            self.assertNotIn("http://", html)
            self.assertNotIn("https://", html)

    def test_per_frame_buffers_are_reset(self):
        """Regression: labels accumulated across frames and painted over the scene."""
        from safesitegen.environment import _TEMPLATE
        draw = _TEMPLATE.split("function draw()")[1].split("function dist(")[0]
        for buf in ("pickable = []", "labels = []"):
            self.assertIn(buf, draw, f"{buf} must be reset inside draw()")

    def test_scene_contract_uses_unity_safe_patrol_shape(self):
        site = SiteModel.load("steel_frame_level3").to_dict()
        self.assertTrue(all(isinstance(p, dict) and "x" in p and "y" in p
                            for p in site["patrol"]))

    def test_answer_key_matches_the_hazards(self):
        from safesitegen.export import to_unity_scene
        result = generate(site_template="utility_trench_corridor", seed=7)
        scene = to_unity_scene(result.scenario,
                               SiteModel.load(result.scenario.site_template), result.report)
        self.assertEqual(len(scene["answerKey"]), len(result.scenario.hazards))
        scored = [i for i in scene["instances"] if i["scoring"] is not None]
        self.assertEqual(len(scored), len(result.scenario.hazards))


if __name__ == "__main__":
    unittest.main()
