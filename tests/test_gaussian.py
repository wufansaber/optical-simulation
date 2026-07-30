import math
import unittest

import numpy as np

from optical_sim import FreeSpace, GaussianBeam, OpticalSystem, ThinLens, sample_gaussian_system


class GaussianBeamTests(unittest.TestCase):
    def setUp(self):
        self.wavelength = 355e-9
        self.waist_radius = 0.5e-3
        self.beam = GaussianBeam.from_waist(
            wavelength=self.wavelength,
            waist_radius=self.waist_radius,
            power=1.0,
        )

    def test_waist_plane_properties(self):
        self.assertAlmostEqual(self.beam.beam_radius, self.waist_radius, places=14)
        self.assertAlmostEqual(self.beam.waist_position(0.0), 0.0, places=14)
        self.assertTrue(math.isinf(self.beam.radius_of_curvature))

    def test_free_space_matches_gaussian_radius_formula(self):
        distance = 0.4
        output = self.beam.propagate(FreeSpace(distance).matrix, distance)
        expected_radius = self.waist_radius * math.sqrt(
            1.0 + (distance / self.beam.rayleigh_range) ** 2
        )
        self.assertAlmostEqual(output.beam_radius, expected_radius, places=14)
        self.assertAlmostEqual(output.waist_position(distance), 0.0, places=13)

    def test_thin_lens_keeps_radius_and_changes_output_waist(self):
        lens_position = 0.2
        focal_length = 0.1
        before_lens = self.beam.propagate(FreeSpace(lens_position).matrix, lens_position)
        after_lens = before_lens.propagate(ThinLens(focal_length).matrix)
        expected_q = 1.0 / (1.0 / before_lens.q - 1.0 / focal_length)

        self.assertAlmostEqual(after_lens.beam_radius, before_lens.beam_radius, places=14)
        self.assertAlmostEqual(after_lens.q.real, expected_q.real, places=14)
        self.assertAlmostEqual(after_lens.q.imag, expected_q.imag, places=14)
        self.assertGreater(after_lens.waist_position(lens_position), lens_position)

    def test_plane_measurement_recovers_waist(self):
        distance = 0.3
        propagated = self.beam.propagate(FreeSpace(distance).matrix, distance)
        reconstructed = GaussianBeam.from_plane(
            wavelength=self.wavelength,
            beam_radius=propagated.beam_radius,
            radius_of_curvature=propagated.radius_of_curvature,
            power=1.0,
        )
        self.assertAlmostEqual(reconstructed.waist_radius, self.waist_radius, places=13)
        self.assertAlmostEqual(reconstructed.waist_position(distance), 0.0, places=12)

    def test_field_is_power_normalized(self):
        expected_peak = 2.0 * self.beam.power / (math.pi * self.waist_radius ** 2)
        np.testing.assert_allclose(self.beam.peak_intensity, expected_peak, rtol=1e-12)
        np.testing.assert_allclose(
            self.beam.intensity(self.waist_radius),
            expected_peak * math.exp(-2.0),
            rtol=1e-12,
        )

    def test_complete_system_returns_stable_waist_position(self):
        system = OpticalSystem([FreeSpace(0.2), ThinLens(0.1), FreeSpace(0.16)])
        points = sample_gaussian_system(system, self.beam, samples_per_space=20)
        output = points[-1]
        waist_position = output.beam.waist_position(output.z)
        self.assertGreater(waist_position, 0.2)
        self.assertLess(waist_position, 0.36)
        self.assertLess(output.beam.waist_radius, 30e-6)


if __name__ == "__main__":
    unittest.main()

