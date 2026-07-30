import unittest

import numpy as np

from optical_sim import FreeSpace, OpticalSystem, Ray, ThinLens


class ParaxialTests(unittest.TestCase):
    def test_free_space_changes_height_from_angle(self):
        output = OpticalSystem([FreeSpace(0.2)]).propagate(Ray(0.01, 0.1))
        self.assertAlmostEqual(output.height, 0.03)
        self.assertAlmostEqual(output.angle, 0.1)

    def test_thin_lens_changes_ray_angle(self):
        output = OpticalSystem([ThinLens(0.1)]).propagate(Ray(0.01, 0.0))
        self.assertAlmostEqual(output.height, 0.01)
        self.assertAlmostEqual(output.angle, -0.1)

    def test_parallel_ray_reaches_back_focal_plane(self):
        system = OpticalSystem([FreeSpace(0.1), ThinLens(0.08), FreeSpace(0.08)])
        output = system.propagate(Ray(0.012, 0.0))
        self.assertAlmostEqual(output.height, 0.0, places=12)

    def test_system_matrix_order(self):
        system = OpticalSystem([FreeSpace(0.1), ThinLens(0.2)])
        expected = ThinLens(0.2).matrix @ FreeSpace(0.1).matrix
        np.testing.assert_allclose(system.matrix, expected)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            FreeSpace(-0.1)
        with self.assertRaises(ValueError):
            ThinLens(0.0)


if __name__ == "__main__":
    unittest.main()

