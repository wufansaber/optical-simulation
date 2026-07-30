import math
import unittest

import numpy as np

from optical_sim.interactive_model import (
    InteractiveConfig,
    InteractiveLens,
    PropagationDirection,
    intensity_on_output_z_plane,
    sample_interactive_path,
    simulate_to_z,
    stripe_355nm_preset,
)


class InteractiveDirectionTests(unittest.TestCase):
    def test_direction_controls_path_length_and_center(self):
        direction = PropagationDirection(theta_x_deg=45.0, theta_y_deg=0.0)
        self.assertAlmostEqual(direction.path_length(0.1), math.sqrt(2.0) * 0.1)
        center_x, center_y = direction.center_at_z(0.1)
        self.assertAlmostEqual(center_x, 0.1)
        self.assertAlmostEqual(center_y, 0.0)
        basis_x, basis_y = direction.transverse_basis
        self.assertAlmostEqual(float(np.dot(direction.vector, basis_x)), 0.0, places=13)
        self.assertAlmostEqual(float(np.dot(direction.vector, basis_y)), 0.0, places=13)

    def test_invalid_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            PropagationDirection(theta_x_deg=89.0)


class InteractivePropagationTests(unittest.TestCase):
    def setUp(self):
        self.config = InteractiveConfig()

    def test_no_lens_matches_free_space(self):
        result = simulate_to_z(self.config, [], 0.2)
        expected = self.config.input_beam().free_space(0.2)
        self.assertAlmostEqual(result.observation.beam.qx.real, expected.qx.real, places=13)
        self.assertAlmostEqual(result.observation.beam.qy.imag, expected.qy.imag, places=13)
        self.assertEqual(len(result.states), 2)

    def test_input_divergence_sets_independent_local_envelope_slopes(self):
        config = InteractiveConfig(
            input_divergence_x_mrad=2.0,
            input_divergence_y_mrad=-1.0,
        )
        beam = config.input_beam()
        radius_x = 0.5 * config.input_diameter_x_m
        radius_y = 0.5 * config.input_diameter_y_m
        self.assertAlmostEqual(
            (1.0 / beam.qx).real,
            math.tan(2.0e-3) / radius_x,
            places=12,
        )
        self.assertAlmostEqual(
            (1.0 / beam.qy).real,
            math.tan(-1.0e-3) / radius_y,
            places=12,
        )

    def test_zero_input_divergence_preserves_waist_plane_default(self):
        beam = InteractiveConfig().input_beam()
        self.assertAlmostEqual((1.0 / beam.qx).real, 0.0, places=14)
        self.assertAlmostEqual((1.0 / beam.qy).real, 0.0, places=14)

    def test_invalid_input_divergence_is_rejected(self):
        with self.assertRaises(ValueError):
            InteractiveConfig(input_divergence_x_mrad=0.5 * math.pi * 1e3)

    def test_xy_focal_lengths_are_independent(self):
        lens = InteractiveLens("LX", 0.1, focal_x_m=0.05, focal_y_m=None)
        result = simulate_to_z(self.config, [lens], 0.1)
        before = self.config.input_beam().free_space(0.1)
        self.assertNotEqual(result.observation.beam.qx, before.qx)
        self.assertEqual(result.observation.beam.qy, before.qy)

    def test_output_before_lens_ignores_that_lens(self):
        lens = InteractiveLens("LATE", 0.3, focal_x_m=0.1, focal_y_m=0.1)
        result = simulate_to_z(self.config, [lens], 0.2)
        self.assertEqual(result.applied_lenses, ())
        self.assertEqual(result.ignored_lenses, (lens,))

    def test_path_sampling_ends_at_requested_plane(self):
        lens = InteractiveLens("L1", 0.1, focal_x_m=0.2, focal_y_m=-0.3)
        samples = sample_interactive_path(self.config, [lens], 0.4, samples_per_segment=10)
        self.assertAlmostEqual(samples[-1].z_m, 0.4, places=13)
        self.assertGreater(len(samples), 10)

    def test_normal_output_plane_matches_beam_intensity(self):
        result = simulate_to_z(self.config, [], 0.1)
        field = intensity_on_output_z_plane(result, grid_points=101, span_radii=2.0)
        x_grid, y_grid = np.meshgrid(field.x_offset_m, field.y_offset_m)
        expected = result.observation.beam.intensity(x_grid, y_grid)
        np.testing.assert_allclose(field.intensity_w_m2, expected, rtol=1e-12)
        self.assertAlmostEqual(float(field.normalized_intensity[50, 50]), 1.0)

    def test_oblique_output_plane_is_finite_and_centered(self):
        config = InteractiveConfig(
            direction=PropagationDirection(theta_x_deg=20.0, theta_y_deg=-12.0)
        )
        result = simulate_to_z(config, [], 0.15)
        field = intensity_on_output_z_plane(result, grid_points=101)
        self.assertTrue(np.all(np.isfinite(field.intensity_w_m2)))
        self.assertAlmostEqual(field.center_x_m, math.tan(math.radians(20.0)) * 0.15)
        self.assertAlmostEqual(field.center_y_m, math.tan(math.radians(-12.0)) * 0.15)

    def test_stripe_preset_reproduces_target_sizes(self):
        result = simulate_to_z(self.config, stripe_355nm_preset(), 272.502149e-3)
        self.assertAlmostEqual(result.observation.beam.radius_x * 2e6, 0.8, places=3)
        self.assertAlmostEqual(result.observation.beam.radius_y * 2e3, 1.5, places=3)


if __name__ == "__main__":
    unittest.main()
