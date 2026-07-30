import tempfile
import unittest
from pathlib import Path

from optical_sim.lens_csv import load_lenses_csv


class LensCsvTests(unittest.TestCase):
    def _load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lenses.csv"
            path.write_text(text, encoding="utf-8-sig")
            return load_lenses_csv(path)

    def test_imports_software_export_format(self):
        result = self._load(
            "name,z_mm,focal_x_mm,focal_y_mm\n"
            "X1,10,-25,\n"
            "Y1,20,,50\n"
        )
        self.assertEqual([lens.name for lens in result.lenses], ["X1", "Y1"])
        self.assertAlmostEqual(result.lenses[0].focal_x_m, -0.025)
        self.assertIsNone(result.lenses[0].focal_y_m)
        self.assertAlmostEqual(result.lenses[1].focal_y_m, 0.05)
        self.assertIsNone(result.observation_z_m)

    def test_imports_optimization_format_and_target(self):
        result = self._load(
            "name,z_mm,focal_x_mm,focal_y_mm,distance_from_previous_mm\n"
            "G1,0,-100,-100,0\n"
            "G2,10,105,105,10\n"
            "TARGET,25,,,15\n"
        )
        self.assertEqual(len(result.lenses), 2)
        self.assertAlmostEqual(result.observation_z_m, 0.025)

    def test_supports_short_header_aliases_and_empty_markers(self):
        result = self._load(
            "name,z,fx,fy\n"
            "L1,5,--,100\n"
            "L2,7,-50,None\n"
        )
        self.assertEqual(len(result.lenses), 2)
        self.assertIsNone(result.lenses[0].focal_x_m)
        self.assertIsNone(result.lenses[1].focal_y_m)

    def test_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "缺少必要列"):
            self._load("name,z_mm\nL1,10\n")

    def test_rejects_invalid_row_with_line_number(self):
        with self.assertRaisesRegex(ValueError, "第 2 行"):
            self._load(
                "name,z_mm,focal_x_mm,focal_y_mm\n"
                "L1,not-a-number,50,\n"
            )

    def test_rejects_file_without_lenses(self):
        with self.assertRaisesRegex(ValueError, "没有可导入"):
            self._load(
                "name,z_mm,focal_x_mm,focal_y_mm\n"
                "TARGET,100,,\n"
            )

    def test_rejects_invalid_target_position(self):
        with self.assertRaisesRegex(ValueError, "TARGET 输出位置"):
            self._load(
                "name,z_mm,focal_x_mm,focal_y_mm\n"
                "L1,0,50,\n"
                "TARGET,-1,,\n"
            )


if __name__ == "__main__":
    unittest.main()
