import math
import unittest

import numpy as np

from optical_sim import (
    AstigmaticGaussianBeam,
    StripeDesignConfig,
    solve_stripe_design,
    surface_intensity,
)


class AstigmaticBeamTests(unittest.TestCase):
    def test_elliptical_field_is_power_normalized(self):
        beam = AstigmaticGaussianBeam.from_plane(
            wavelength=355e-9,
            radius_x=0.4e-3,
            radius_y=0.8e-3,
            power=2.0,
        )
        expected_peak = 2.0 * beam.power / (math.pi * beam.radius_x * beam.radius_y)
        np.testing.assert_allclose(beam.peak_intensity, expected_peak, rtol=1e-12)
        np.testing.assert_allclose(
            beam.intensity(beam.radius_x, 0.0),
            expected_peak * math.exp(-2.0),
            rtol=1e-12,
        )

    def test_x_cylindrical_lens_does_not_change_y_q(self):
        beam = AstigmaticGaussianBeam.from_plane(355e-9, 0.6e-3, 0.6e-3)
        output = beam.apply_lens(focal_x=25e-3)
        self.assertNotEqual(output.qx, beam.qx)
        self.assertEqual(output.qy, beam.qy)
        self.assertAlmostEqual(output.radius_x, beam.radius_x, places=14)
        self.assertAlmostEqual(output.radius_y, beam.radius_y, places=14)


class StripeDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = solve_stripe_design(StripeDesignConfig())

    def test_target_surface_diameters(self):
        self.assertAlmostEqual(self.design.surface_diameter_x * 1e6, 0.8, places=8)
        self.assertAlmostEqual(self.design.surface_diameter_y * 1e3, 1.5, places=8)

    def test_solved_focal_lengths(self):
        self.assertAlmostEqual(self.design.x_positive_focal * 1e3, 168.1531, places=3)
        self.assertAlmostEqual(self.design.y_positive_focal * 1e3, 59.3490, places=3)
        self.assertAlmostEqual(self.design.focus_distance * 1e3, 25.0000, places=3)

    def test_prescription_is_ordered(self):
        positions = [lens.z for lens in self.design.prescription]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(self.design.prescription), 7)
        self.assertAlmostEqual(self.design.surface_z * 1e3, 272.5021, places=3)

    def test_incidence_angle_is_defined_to_surface_normal(self):
        self.assertAlmostEqual(self.design.config.incidence_angle_to_normal_deg, 0.0)
        self.assertAlmostEqual(self.design.incidence_angle_to_plane_deg, 90.0)


    def test_normal_incidence_solution_is_within_model_slope_limit(self):
        self.assertAlmostEqual(self.design.paraxial_slope_x, 0.2825000, places=6)
        self.assertAlmostEqual(self.design.geometric_na_x, 0.2718602, places=6)
        self.assertTrue(self.design.paraxial_model_valid_x)

    def test_surface_center_is_peak(self):
        center = float(surface_intensity(self.design, 0.0, 0.0))
        edge_x = float(
            surface_intensity(self.design, self.design.surface_x_positive, 0.0)
        )
        edge_y = float(
            surface_intensity(self.design, 0.0, self.design.surface_diameter_y / 2.0)
        )
        self.assertGreater(center, edge_x)
        np.testing.assert_allclose(edge_x / center, math.exp(-2.0), rtol=1e-10)
        np.testing.assert_allclose(edge_y / center, math.exp(-2.0), rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
