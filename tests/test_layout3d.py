import math
import unittest

import numpy as np

from optical_sim.layout3d import default_layout_3d, validate_layout_3d


class Layout3DTests(unittest.TestCase):
    def setUp(self):
        self.layout = default_layout_3d()

    def test_both_illumination_paths_close_at_180_mm(self):
        validate_layout_3d(self.layout)
        for path in (self.layout.near_path, self.layout.oblique_path):
            self.assertEqual(path[0].s_mm, 0.0)
            self.assertEqual(path[-1].s_mm, 180.0)
            self.assertEqual(path[-1].position_mm, (0.0, 0.0, 0.0))

    def test_final_directions_match_incidence_angles(self):
        wafer_normal = np.asarray(self.layout.wafer_normal)
        for direction, expected in (
            (self.layout.near_direction, 1.0),
            (self.layout.oblique_direction, 75.0),
        ):
            direction = np.asarray(direction)
            angle = math.degrees(
                math.acos(np.clip(np.dot(-direction, wafer_normal), -1.0, 1.0))
            )
            self.assertAlmostEqual(angle, expected, places=9)

    def test_objective_is_six_mm_from_wafer(self):
        objective = next(
            element for element in self.layout.elements if element.element_id == "OBJECTIVE_IDEAL"
        )
        distance = np.linalg.norm(np.asarray(objective.position_mm))
        self.assertAlmostEqual(distance, 6.0, places=9)
        self.assertIn("30 mm", objective.note)

    def test_common_path_is_identical_to_prism(self):
        near = [vertex for vertex in self.layout.near_path if vertex.s_mm <= 100.0]
        oblique = [vertex for vertex in self.layout.oblique_path if vertex.s_mm <= 100.0]
        near_geometry = [(vertex.element_id, vertex.s_mm, vertex.position_mm) for vertex in near]
        oblique_geometry = [(vertex.element_id, vertex.s_mm, vertex.position_mm) for vertex in oblique]
        self.assertEqual(near_geometry, oblique_geometry)

    def test_all_mirror_normals_are_unit_vectors(self):
        mirrors = [
            element for element in self.layout.elements if element.component_type == "平面反射镜"
        ]
        self.assertGreaterEqual(len(mirrors), 7)
        for mirror in mirrors:
            self.assertAlmostEqual(np.linalg.norm(mirror.surface_normal), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
