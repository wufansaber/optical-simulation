import unittest

from optical_sim.dual_branch import (
    BranchRequirement,
    Paladin355Input,
    default_dual_branch_requirements,
    default_estimated_layout,
    sample_branch_envelope,
    sample_paraxial_ray_bundle,
    solve_equivalent_branch,
)


class DualBranchInputTests(unittest.TestCase):
    def test_default_matrix_has_two_angles_and_two_ratios(self):
        requirements = default_dual_branch_requirements()
        self.assertEqual(len(requirements), 4)
        self.assertEqual(
            [item.incidence_angle_to_normal_deg for item in requirements],
            [1.0, 1.0, 75.0, 75.0],
        )
        for actual, expected in zip(
            [item.target_surface_diameter_x_m * 1e6 for item in requirements],
            [1.44, 1.05, 1.44, 1.05],
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            [item.target_surface_diameter_y_m * 1e3 for item in requirements],
            [5.2, 2.667, 5.2, 2.667],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(item.total_path_m == 0.18 for item in requirements))
        for requirement in requirements[:2]:
            self.assertEqual(requirement.lens2_s_m, 0.166)
            self.assertEqual(requirement.objective_s_m, 0.174)
            self.assertEqual(requirement.objective_focal_m, 0.030)
        for requirement in requirements[2:]:
            self.assertIsNone(requirement.objective_s_m)
            self.assertIsNone(requirement.objective_focal_m)

    def test_invalid_lens_order_is_rejected(self):
        with self.assertRaises(ValueError):
            BranchRequirement("bad", 1.0, 1e-6, 1e-3, lens1_s_m=0.17, lens2_s_m=0.16)

    def test_estimated_layout_closes_at_180_mm(self):
        layout = default_estimated_layout()
        wafer_positions = [item.s_m for item in layout if item.element == "WAFER"]
        self.assertEqual(len(wafer_positions), 2)
        self.assertTrue(all(position == 0.18 for position in wafer_positions))
        self.assertTrue(any(item.element == "OBLIQUE_LFC_E2" and item.s_m == 0.174 for item in layout))
        self.assertTrue(any(item.element == "OBJECTIVE_IDEAL" and item.s_m == 0.174 for item in layout))


class DualBranchSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.laser = Paladin355Input()
        cls.results = [
            solve_equivalent_branch(requirement, cls.laser)
            for requirement in default_dual_branch_requirements()
        ]

    def test_all_cases_hit_surface_targets(self):
        for result in self.results:
            requirement = result.requirement
            self.assertAlmostEqual(
                result.output_surface_diameter_x_m,
                requirement.target_surface_diameter_x_m,
                delta=requirement.target_surface_diameter_x_m * 1e-7,
            )
            self.assertAlmostEqual(
                result.output_surface_diameter_y_m,
                requirement.target_surface_diameter_y_m,
                delta=requirement.target_surface_diameter_y_m * 1e-7,
            )

    def test_equivalent_lens_planes_are_inside_path(self):
        for result in self.results:
            lenses = result.lenses
            self.assertGreater(lenses.lens1_s_m, 0.0)
            self.assertGreater(lenses.lens2_s_m, lenses.lens1_s_m)
            self.assertLess(lenses.lens2_s_m, result.requirement.total_path_m)

    def test_oblique_x_focus_reports_paraxial_warning(self):
        oblique_results = [
            result
            for result in self.results
            if result.requirement.incidence_angle_to_normal_deg == 75.0
        ]
        self.assertEqual(len(oblique_results), 2)
        self.assertTrue(all(not result.paraxial_valid_x for result in oblique_results))
        focal_lengths = [result.lenses.focal2_x_m for result in oblique_results]
        self.assertLess(abs(focal_lengths[0] - focal_lengths[1]), 0.05e-3)

    def test_sampled_path_ends_at_180_mm(self):
        for result in self.results:
            s, radius_x, radius_y = sample_branch_envelope(result, samples_per_segment=10)
            self.assertAlmostEqual(s[-1], 0.18)
            self.assertGreater(radius_x.min(), 0.0)
            self.assertGreater(radius_y.min(), 0.0)

    def test_paraxial_ray_bundle_matches_gaussian_wafer_radius(self):
        for result in self.results:
            for axis, expected_diameter in (
                ("x", result.output_normal_diameter_x_m),
                ("y", result.output_normal_diameter_y_m),
            ):
                s, rays = sample_paraxial_ray_bundle(
                    result,
                    axis,
                    ray_count=65,
                    samples_per_segment=10,
                )
                self.assertAlmostEqual(s[-1], 0.18)
                self.assertEqual(rays.shape, (len(s), 65))
                self.assertAlmostEqual(
                    2.0 * max(abs(rays[-1])),
                    expected_diameter,
                    delta=expected_diameter * 0.01,
                )

    def test_invalid_ray_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_paraxial_ray_bundle(self.results[0], "z")


if __name__ == "__main__":
    unittest.main()
