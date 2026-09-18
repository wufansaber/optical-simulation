import unittest

from optical_sim import (
    IlluminationPerformanceRequirement,
    default_dual_branch_requirements,
    default_efficiency_budget,
    evaluate_case_performance,
    minimum_supergaussian_order,
    supergaussian_capture_fraction,
    supergaussian_edge_uniformity,
)


class UniformityModelTests(unittest.TestCase):
    def test_order_36_has_margin_over_90_percent(self):
        uniformity = supergaussian_edge_uniformity(36.0, 0.90)
        self.assertGreater(uniformity, 0.95)

    def test_minimum_order_is_below_design_order(self):
        minimum = minimum_supergaussian_order(0.90, 0.90)
        self.assertGreater(minimum, 27.0)
        self.assertLess(minimum, 29.0)
        self.assertLess(minimum, 36.0)

    def test_target_capture_is_dominated_by_gaussian_x(self):
        capture = supergaussian_capture_fraction(36.0)
        self.assertGreater(capture, 0.998)
        self.assertLessEqual(capture, 1.0)


class PerformanceBudgetTests(unittest.TestCase):
    def test_all_four_cases_meet_allocated_requirements(self):
        performance = IlluminationPerformanceRequirement()
        results = [
            evaluate_case_performance(requirement, performance)
            for requirement in default_dual_branch_requirements()
        ]
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertTrue(result.uniformity_pass, result.case_name)
            self.assertTrue(result.efficiency_pass, result.case_name)
            self.assertGreaterEqual(result.uniformity_allocated_minimum_ratio, 0.90)
            self.assertGreaterEqual(result.useful_efficiency_allocated_minimum_ratio, 0.65)

    def test_ratio_b_has_lower_fibe_budget_than_ratio_a(self):
        requirements = default_dual_branch_requirements()
        ratio_a = next(item for item in requirements if item.name == "near_normal_1deg_ratio_a")
        ratio_b = next(item for item in requirements if item.name == "near_normal_1deg_ratio_b")
        budget_a = default_efficiency_budget(ratio_a)
        budget_b = default_efficiency_budget(ratio_b)
        fibe_a = next(item for item in budget_a if item.group == "MAGNIFICATION_SWITCH")
        fibe_b = next(item for item in budget_b if item.group == "MAGNIFICATION_SWITCH")
        self.assertGreater(fibe_a.nominal_ratio, fibe_b.nominal_ratio)

    def test_invalid_requirement_is_rejected(self):
        with self.assertRaises(ValueError):
            IlluminationPerformanceRequirement(minimum_uniformity_ratio=1.1)


if __name__ == "__main__":
    unittest.main()
