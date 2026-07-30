import math
import unittest

from optical_sim import (
    StripeDesignConfig,
    lens_plane_states,
    sample_optical_path,
    solve_stripe_design,
)


class LensPlaneOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = solve_stripe_design(StripeDesignConfig())
        cls.states = lens_plane_states(cls.design)

    def test_all_optical_element_planes_are_returned(self):
        self.assertEqual(
            [state.lens.name for state in self.states],
            ["X1", "X2", "Y1", "Y2", "C1", "C2", "F"],
        )
        self.assertEqual(len(self.states), 7)


    def test_last_lens_propagates_to_saved_surface_state(self):
        last_state = self.states[-1]
        surface = last_state.beam_after.free_space(last_state.lens.distance_to_next)
        self.assertAlmostEqual(surface.qx.real, self.design.beam_at_surface.qx.real, places=13)
        self.assertAlmostEqual(surface.qx.imag, self.design.beam_at_surface.qx.imag, places=13)
        self.assertAlmostEqual(surface.qy.real, self.design.beam_at_surface.qy.real, places=12)
        self.assertAlmostEqual(surface.qy.imag, self.design.beam_at_surface.qy.imag, places=12)

    def test_first_plane_has_expected_one_over_e2_field(self):
        beam = self.states[0].beam_after
        center = float(beam.intensity(0.0, 0.0))
        edge = float(beam.intensity(beam.radius_x, 0.0))
        self.assertAlmostEqual(edge / center, math.exp(-2.0), places=12)

    def test_optical_path_ends_at_target(self):
        samples = sample_optical_path(self.design, samples_per_segment=10)
        self.assertAlmostEqual(samples[-1].z, self.design.surface_z, places=13)
        self.assertAlmostEqual(samples[-1].radius_x, self.design.beam_at_surface.radius_x, places=13)
        self.assertAlmostEqual(samples[-1].radius_y, self.design.beam_at_surface.radius_y, places=13)


if __name__ == "__main__":
    unittest.main()
