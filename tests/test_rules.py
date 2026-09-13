"""Regulatory predicate evaluation: the part that has to be exactly right."""

import unittest

from safesitegen.knowledge import RULES_BY_ID
from safesitegen.schema import Entity, Hazard, Relation, Scenario, SchemaError
from safesitegen.validate import COMPLIANT, NOT_APPLICABLE, VIOLATION, audit, evaluate_rule


def worker(context, **params):
    p = {"context": context, "controls": params.pop("controls", [])}
    p.update(params)
    return Entity(id="w1", kind="worker", asset_type="labourer", params=p)


class TestThresholdRules(unittest.TestCase):
    def test_edge_below_trigger_height_is_compliant(self):
        rule = RULES_BY_ID["1926.501(b)(1)"]
        e = worker("edge_exposure", fall_height_ft=4.0)
        self.assertEqual(evaluate_rule(rule, e), COMPLIANT)

    def test_edge_at_trigger_height_without_control_violates(self):
        rule = RULES_BY_ID["1926.501(b)(1)"]
        e = worker("edge_exposure", fall_height_ft=6.0)
        self.assertEqual(evaluate_rule(rule, e), VIOLATION)

    def test_edge_with_any_accepted_control_is_compliant(self):
        rule = RULES_BY_ID["1926.501(b)(1)"]
        for control in ("guardrail_system", "safety_net", "personal_fall_arrest"):
            e = worker("edge_exposure", fall_height_ft=30.0, controls=[control])
            self.assertEqual(evaluate_rule(rule, e), COMPLIANT, control)

    def test_unrelated_control_does_not_satisfy_the_clause(self):
        rule = RULES_BY_ID["1926.501(b)(1)"]
        e = worker("edge_exposure", fall_height_ft=30.0, controls=["hard_hat"])
        self.assertEqual(evaluate_rule(rule, e), VIOLATION)

    def test_missing_parameter_is_not_applicable(self):
        rule = RULES_BY_ID["1926.501(b)(1)"]
        e = worker("edge_exposure")
        self.assertEqual(evaluate_rule(rule, e), NOT_APPLICABLE)

    def test_wrong_context_is_not_applicable(self):
        rule = RULES_BY_ID["1926.501(b)(1)"]
        e = worker("scaffold_platform", fall_height_ft=30.0)
        self.assertEqual(evaluate_rule(rule, e), NOT_APPLICABLE)

    def test_scaffold_uses_strict_greater_than(self):
        rule = RULES_BY_ID["1926.451(g)(1)"]
        at_limit = worker("scaffold_platform", platform_height_ft=10.0)
        above = worker("scaffold_platform", platform_height_ft=10.5)
        self.assertEqual(evaluate_rule(rule, at_limit), COMPLIANT)
        self.assertEqual(evaluate_rule(rule, above), VIOLATION)


class TestOtherPredicateKinds(unittest.TestCase):
    def test_guardrail_height_range(self):
        rule = RULES_BY_ID["1926.502(b)(1)"]
        for h, expected in ((42.0, COMPLIANT), (39.0, COMPLIANT), (45.0, COMPLIANT),
                            (33.0, VIOLATION), (48.0, VIOLATION)):
            e = Entity(id="g", kind="control", asset_type="guardrail_system",
                       params={"top_rail_height_in": h})
            self.assertEqual(evaluate_rule(rule, e), expected, h)

    def test_ladder_extension_or_secured_top(self):
        rule = RULES_BY_ID["1926.1053(b)(1)"]
        short = Entity(id="l", kind="equipment", asset_type="portable_ladder",
                       params={"extension_above_landing_ft": 1.0, "controls": []})
        secured = Entity(id="l", kind="equipment", asset_type="portable_ladder",
                         params={"extension_above_landing_ft": 1.0,
                                 "controls": ["secured_top_and_grabrail"]})
        long = Entity(id="l", kind="equipment", asset_type="portable_ladder",
                      params={"extension_above_landing_ft": 3.5, "controls": []})
        self.assertEqual(evaluate_rule(rule, short), VIOLATION)
        self.assertEqual(evaluate_rule(rule, secured), COMPLIANT)
        self.assertEqual(evaluate_rule(rule, long), COMPLIANT)

    def test_excavation_stable_rock_exemption(self):
        rule = RULES_BY_ID["1926.652(a)(1)"]
        deep = worker("excavation", excavation_depth_ft=9.0, soil_type="type_c")
        rock = worker("excavation", excavation_depth_ft=9.0, soil_type="stable_rock")
        shored = worker("excavation", excavation_depth_ft=9.0, soil_type="type_c",
                        controls=["trench_shield"])
        self.assertEqual(evaluate_rule(rule, deep), VIOLATION)
        self.assertEqual(evaluate_rule(rule, rock), COMPLIANT)
        self.assertEqual(evaluate_rule(rule, shored), COMPLIANT)

    def test_lateral_travel_only_applies_past_depth_trigger(self):
        rule = RULES_BY_ID["1926.651(c)(2)"]
        shallow = worker("excavation", excavation_depth_ft=3.0, egress_lateral_travel_ft=90.0)
        deep_far = worker("excavation", excavation_depth_ft=6.0, egress_lateral_travel_ft=40.0)
        deep_near = worker("excavation", excavation_depth_ft=6.0, egress_lateral_travel_ft=12.0)
        self.assertEqual(evaluate_rule(rule, shallow), COMPLIANT)
        self.assertEqual(evaluate_rule(rule, deep_far), VIOLATION)
        self.assertEqual(evaluate_rule(rule, deep_near), COMPLIANT)

    def test_requires_any_control(self):
        rule = RULES_BY_ID["1926.601(b)(4)"]
        bare = Entity(id="t", kind="equipment", asset_type="haul_truck", params={"controls": []})
        alarmed = Entity(id="t", kind="equipment", asset_type="haul_truck",
                         params={"controls": ["reverse_alarm"]})
        self.assertEqual(evaluate_rule(rule, bare), VIOLATION)
        self.assertEqual(evaluate_rule(rule, alarmed), COMPLIANT)

    def test_crane_clearance_or_deenergised(self):
        rule = RULES_BY_ID["1926.1408(a)"]
        close = Entity(id="c", kind="equipment", asset_type="mobile_crane",
                       params={"clearance_to_line_ft": 8.0, "controls": []})
        clear = Entity(id="c", kind="equipment", asset_type="mobile_crane",
                       params={"clearance_to_line_ft": 25.0, "controls": []})
        isolated = Entity(id="c", kind="equipment", asset_type="mobile_crane",
                          params={"clearance_to_line_ft": 8.0,
                                  "controls": ["line_deenergised_and_grounded"]})
        self.assertEqual(evaluate_rule(rule, close), VIOLATION)
        self.assertEqual(evaluate_rule(rule, clear), COMPLIANT)
        self.assertEqual(evaluate_rule(rule, isolated), COMPLIANT)


class TestAudit(unittest.TestCase):
    def test_audit_finds_every_violating_entity(self):
        s = Scenario(id="t", site_template="steel_frame_level3", trade="x", activity="y")
        s.entities = [
            worker("edge_exposure", fall_height_ft=20.0),
            Entity(id="g", kind="control", asset_type="guardrail_system",
                   params={"top_rail_height_in": 30.0}),
            Entity(id="ok", kind="control", asset_type="guardrail_system",
                   params={"top_rail_height_in": 42.0}),
        ]
        result = audit(s)
        self.assertIn("w1", result)
        self.assertIn("g", result)
        self.assertNotIn("ok", result)


class TestSchema(unittest.TestCase):
    def test_unknown_entity_kind_rejected(self):
        s = Scenario(id="t", site_template="steel_frame_level3", trade="a", activity="b")
        s.entities = [Entity(id="e", kind="ghost", asset_type="x")]
        with self.assertRaises(SchemaError):
            s.validate()

    def test_duplicate_entity_id_rejected(self):
        s = Scenario(id="t", site_template="steel_frame_level3", trade="a", activity="b")
        s.entities = [Entity(id="e", kind="worker", asset_type="x"),
                      Entity(id="e", kind="worker", asset_type="y")]
        with self.assertRaises(SchemaError):
            s.validate()

    def test_relation_to_unknown_entity_rejected(self):
        s = Scenario(id="t", site_template="steel_frame_level3", trade="a", activity="b")
        s.entities = [Entity(id="e", kind="worker", asset_type="x")]
        s.relations = [Relation(src="e", dst="nope", type="exposed_to")]
        with self.assertRaises(SchemaError):
            s.validate()

    def test_unknown_relation_type_rejected(self):
        s = Scenario(id="t", site_template="steel_frame_level3", trade="a", activity="b")
        s.entities = [Entity(id="a", kind="worker", asset_type="x"),
                      Entity(id="b", kind="worker", asset_type="y")]
        s.relations = [Relation(src="a", dst="b", type="teleports_to")]
        with self.assertRaises(SchemaError):
            s.validate()

    def test_present_controls_must_be_subset_of_required(self):
        s = Scenario(id="t", site_template="steel_frame_level3", trade="a", activity="b")
        s.entities = [Entity(id="e", kind="worker", asset_type="x")]
        s.hazards = [Hazard(id="h1", hazard_class="fall_unprotected_edge", target_entity="e",
                            clause="1926.501(b)(1)", energy_source="gravitational",
                            required_controls=["guardrail_system"],
                            present_controls=["banana"])]
        with self.assertRaises(SchemaError):
            s.validate()

    def test_roundtrip_serialisation(self):
        s = Scenario(id="t", site_template="steel_frame_level3", trade="a", activity="b")
        s.entities = [Entity(id="e", kind="worker", asset_type="x", x=1.0, y=2.0)]
        clone = Scenario.from_dict(s.to_dict())
        self.assertEqual(clone.entity("e").position(), (1.0, 2.0))
        self.assertEqual(clone.id, s.id)

    def test_signature_is_layout_sensitive(self):
        def make(x):
            s = Scenario(id="t", site_template="steel_frame_level3", trade="a", activity="b")
            s.entities = [Entity(id="e", kind="worker", asset_type="x", x=x, y=2.0)]
            s.hazards = [Hazard(id="h1", hazard_class="fall_unprotected_edge", target_entity="e",
                                clause="1926.501(b)(1)", energy_source="gravitational")]
            return s
        self.assertNotEqual(make(1.0).configuration_signature(),
                            make(9.0).configuration_signature())
        self.assertEqual(make(1.0).configuration_signature(),
                         make(1.0).configuration_signature())


if __name__ == "__main__":
    unittest.main()
