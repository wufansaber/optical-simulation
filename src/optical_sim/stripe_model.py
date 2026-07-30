"""355 nm 柱面扩束、柱面聚焦和 75 度倾斜平面条形光斑设计。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import atan, cos, degrees, exp, isfinite, pi, radians, sin, sqrt
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .astigmatic import AstigmaticGaussianBeam
from .gaussian import SPEED_OF_LIGHT, VACUUM_PERMITTIVITY


@dataclass(frozen=True)
class StripeDesignConfig:
    wavelength: float = 355e-9
    input_diameter: float = 1.2e-3
    target_surface_diameter_x: float = 0.8e-6
    target_surface_diameter_y: float = 1.5e-3
    incidence_angle_to_normal_deg: float = 0.0
    power: float = 1.0
    x_negative_focal: float = -15e-3
    y_negative_focal: float = -50e-3
    common_negative_focal: float = -100e-3
    common_positive_focal: float = 105e-3
    focusing_focal_x: float = 25e-3
    gap_x_to_y: float = 30e-3
    gap_y_to_common: float = 30e-3
    gap_common_to_focus: float = 20e-3

    def __post_init__(self) -> None:
        positive_values = (
            self.wavelength,
            self.input_diameter,
            self.target_surface_diameter_x,
            self.target_surface_diameter_y,
            self.power,
            self.common_positive_focal,
            self.focusing_focal_x,
        )
        if any(not isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError("波长、尺寸、功率及正焦距必须为正的有限数值。")
        if not 0.0 <= self.incidence_angle_to_normal_deg < 90.0:
            raise ValueError("光束与平面法线的夹角必须在 0 到 90 度之间。")
        if self.x_negative_focal >= 0 or self.y_negative_focal >= 0 or self.common_negative_focal >= 0:
            raise ValueError("三个 Galilean 扩束镜的第一片必须是负焦距。")


@dataclass(frozen=True)
class LensPrescription:
    index: int
    name: str
    lens_type: str
    z: float
    focal_x: Optional[float]
    focal_y: Optional[float]
    distance_to_next: float
    incoming_diameter_x: float
    incoming_diameter_y: float
    surface_footprint_x: float
    surface_footprint_y: float
    recommended_clear_x: float
    recommended_clear_y: float


@dataclass(frozen=True)
class StripeDesign:
    config: StripeDesignConfig
    x_positive_focal: float
    y_positive_focal: float
    focus_distance: float
    surface_z: float
    beam_at_surface: AstigmaticGaussianBeam
    surface_x_negative: float
    surface_x_positive: float
    surface_diameter_x: float
    surface_diameter_y: float
    prescription: List[LensPrescription]

    @property
    def incidence_angle_to_normal_rad(self) -> float:
        return radians(self.config.incidence_angle_to_normal_deg)

    @property
    def incidence_angle_to_plane_deg(self) -> float:
        return 90.0 - self.config.incidence_angle_to_normal_deg

    @property
    def incidence_angle_to_plane_rad(self) -> float:
        return radians(self.incidence_angle_to_plane_deg)

    @property
    def paraxial_slope_x(self) -> float:
        """Return the small-angle Gaussian divergence parameter lambda/(pi*w0).

        This equals numerical aperture only in the paraxial limit.  Values near
        or above one are a validity warning, not a physical NA greater than one.
        """

        return self.config.wavelength / (pi * self.beam_at_surface.waist_radius_x)

    @property
    def numerical_aperture_x(self) -> float:
        """Backward-compatible alias for the paraxial divergence parameter."""

        return self.paraxial_slope_x

    @property
    def equivalent_cone_half_angle_deg(self) -> float:
        return degrees(atan(self.paraxial_slope_x))

    @property
    def geometric_na_x(self) -> float:
        slope = self.paraxial_slope_x
        return self.beam_at_surface.refractive_index * slope / sqrt(1.0 + slope ** 2)

    @property
    def paraxial_model_valid_x(self) -> bool:
        return self.paraxial_slope_x <= 0.3

def surface_field(design: StripeDesign, x_surface, y_surface):
    """计算倾斜平面上的复电场。

    入射面为 X-Z，柱面镜轴为 Y。平面坐标 x 对应横向分量
    x*sin(alpha) 和轴向分量 x*cos(alpha)，因此焦线沿 Y 与平面重合。
    """

    beam = design.beam_at_surface
    x_array = np.asarray(x_surface, dtype=np.float64)
    y_array = np.asarray(y_surface, dtype=np.float64)
    axial_offset = x_array * sin(design.incidence_angle_to_normal_rad)
    transverse_x = x_array * cos(design.incidence_angle_to_normal_rad)
    qx = beam.qx + axial_offset
    qy = beam.qy + axial_offset
    diffraction_scale = np.sqrt(
        (1.0 + axial_offset / beam.qx) * (1.0 + axial_offset / beam.qy)
    )
    axis_field = (
        beam.axis_field
        * np.exp(1j * beam.wave_number * axial_offset)
        / diffraction_scale
    )
    transverse_phase = -1j * beam.wave_number * (
        transverse_x ** 2 / (2.0 * qx) + y_array ** 2 / (2.0 * qy)
    )
    return axis_field * np.exp(transverse_phase)


def surface_intensity(design: StripeDesign, x_surface, y_surface):
    electric_field = surface_field(design, x_surface, y_surface)
    return (
        0.5
        * design.beam_at_surface.refractive_index
        * VACUUM_PERMITTIVITY
        * SPEED_OF_LIGHT
        * np.abs(electric_field) ** 2
    )


def _surface_ratio_x(beam: AstigmaticGaussianBeam, config: StripeDesignConfig, x_value: float) -> float:
    angle_to_normal = radians(config.incidence_angle_to_normal_deg)
    axial_offset = x_value * sin(angle_to_normal)
    transverse_x = x_value * cos(angle_to_normal)
    qx = beam.qx + axial_offset
    qy = beam.qy + axial_offset
    scale_squared = 1.0 / abs(
        (1.0 + axial_offset / beam.qx) * (1.0 + axial_offset / beam.qy)
    )
    exponent = -1j * beam.wave_number * transverse_x ** 2 / (2.0 * qx)
    return scale_squared * exp(2.0 * exponent.real)


def _one_over_e2_crossing(
    beam: AstigmaticGaussianBeam,
    config: StripeDesignConfig,
    direction: float,
) -> float:
    target = exp(-2.0)
    low = 0.0
    high = config.target_surface_diameter_x
    while _surface_ratio_x(beam, config, direction * high) > target:
        high *= 2.0
        if high > 1e-3:
            raise RuntimeError("无法找到 X 方向的 1/e^2 交点。")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if _surface_ratio_x(beam, config, direction * middle) > target:
            low = middle
        else:
            high = middle
    return direction * 0.5 * (low + high)


def _simulate(
    config: StripeDesignConfig,
    x_positive_focal: float,
    y_positive_focal: float,
    collect_prescription: bool = False,
):
    beam = AstigmaticGaussianBeam.from_plane(
        wavelength=config.wavelength,
        radius_x=config.input_diameter / 2.0,
        radius_y=config.input_diameter / 2.0,
        power=config.power,
    )
    z_position = 0.0
    lens_records = []

    def record(name, lens_type, focal_x, focal_y):
        if collect_prescription:
            lens_records.append(
                (
                    name,
                    lens_type,
                    z_position,
                    focal_x,
                    focal_y,
                    2.0 * beam.radius_x,
                    2.0 * beam.radius_y,
                )
            )

    record("X1", "X cylindrical, diverging", config.x_negative_focal, None)
    beam = beam.apply_lens(focal_x=config.x_negative_focal)
    x_separation = config.x_negative_focal + x_positive_focal
    beam = beam.free_space(x_separation)
    z_position += x_separation
    record("X2", "X cylindrical, collimating", x_positive_focal, None)
    beam = beam.apply_lens(focal_x=x_positive_focal)

    beam = beam.free_space(config.gap_x_to_y)
    z_position += config.gap_x_to_y
    record("Y1", "Y cylindrical, diverging", None, config.y_negative_focal)
    beam = beam.apply_lens(focal_y=config.y_negative_focal)
    y_separation = config.y_negative_focal + y_positive_focal
    beam = beam.free_space(y_separation)
    z_position += y_separation
    record("Y2", "Y cylindrical, collimating", None, y_positive_focal)
    beam = beam.apply_lens(focal_y=y_positive_focal)

    beam = beam.free_space(config.gap_y_to_common)
    z_position += config.gap_y_to_common
    record(
        "C1",
        "Spherical, common diverging",
        config.common_negative_focal,
        config.common_negative_focal,
    )
    beam = beam.apply_lens(
        focal_x=config.common_negative_focal,
        focal_y=config.common_negative_focal,
    )
    common_separation = config.common_negative_focal + config.common_positive_focal
    beam = beam.free_space(common_separation)
    z_position += common_separation
    record(
        "C2",
        "Spherical, common collimating",
        config.common_positive_focal,
        config.common_positive_focal,
    )
    beam = beam.apply_lens(
        focal_x=config.common_positive_focal,
        focal_y=config.common_positive_focal,
    )

    beam = beam.free_space(config.gap_common_to_focus)
    z_position += config.gap_common_to_focus
    record("F", "X cylindrical, focusing", config.focusing_focal_x, None)
    beam = beam.apply_lens(focal_x=config.focusing_focal_x)
    focus_distance = -beam.qx.real
    if focus_distance <= 0:
        raise RuntimeError("计算得到的 X 束腰不在聚焦柱面镜之后。")
    beam = beam.free_space(focus_distance)
    z_position += focus_distance
    return beam, z_position, focus_distance, lens_records


def _measure(
    config: StripeDesignConfig,
    x_positive_focal: float,
    y_positive_focal: float,
):
    beam, surface_z, focus_distance, records = _simulate(
        config,
        x_positive_focal,
        y_positive_focal,
        collect_prescription=False,
    )
    x_negative = _one_over_e2_crossing(beam, config, -1.0)
    x_positive = _one_over_e2_crossing(beam, config, 1.0)
    diameter_x = x_positive - x_negative
    diameter_y = 2.0 * beam.radius_y
    return diameter_x, diameter_y, beam, surface_z, focus_distance, records


def _bisect_focal(
    config: StripeDesignConfig,
    target_index: int,
    target_value: float,
    low: float,
    high: float,
    fixed_focal: float,
    solve_x: bool,
) -> float:
    def residual(parameter):
        x_focal, y_focal = (parameter, fixed_focal) if solve_x else (fixed_focal, parameter)
        return _measure(config, x_focal, y_focal)[target_index] - target_value

    low_residual = residual(low)
    high_residual = residual(high)
    if low_residual * high_residual > 0:
        raise RuntimeError("扩束镜焦距求解区间没有包围目标值。")
    for _ in range(70):
        middle = 0.5 * (low + high)
        middle_residual = residual(middle)
        if low_residual * middle_residual <= 0:
            high = middle
            high_residual = middle_residual
        else:
            low = middle
            low_residual = middle_residual
    return 0.5 * (low + high)


def solve_stripe_design(config: Optional[StripeDesignConfig] = None) -> StripeDesign:
    """求解两组柱面扩束镜的正焦距，使表面尺寸命中目标。"""

    config = config or StripeDesignConfig()
    x_positive_focal = 175e-3
    y_positive_focal = 60e-3
    for _ in range(7):
        x_positive_focal = _bisect_focal(
            config,
            target_index=0,
            target_value=config.target_surface_diameter_x,
            low=16e-3,
            high=2.0,
            fixed_focal=y_positive_focal,
            solve_x=True,
        )
        y_positive_focal = _bisect_focal(
            config,
            target_index=1,
            target_value=config.target_surface_diameter_y,
            low=52e-3,
            high=90e-3,
            fixed_focal=x_positive_focal,
            solve_x=False,
        )

    diameter_x, diameter_y, beam, surface_z, focus_distance, _ = _measure(
        config, x_positive_focal, y_positive_focal
    )
    x_negative = _one_over_e2_crossing(beam, config, -1.0)
    x_positive = _one_over_e2_crossing(beam, config, 1.0)
    _, _, _, lens_records = _simulate(
        config,
        x_positive_focal,
        y_positive_focal,
        collect_prescription=True,
    )

    positions = [record[2] for record in lens_records] + [surface_z]
    prescription = []
    for index, record in enumerate(lens_records):
        name, lens_type, z, focal_x, focal_y, diameter_in_x, diameter_in_y = record
        surface_footprint_x = diameter_in_x
        surface_footprint_y = diameter_in_y
        prescription.append(
            LensPrescription(
                index=index + 1,
                name=name,
                lens_type=lens_type,
                z=z,
                focal_x=focal_x,
                focal_y=focal_y,
                distance_to_next=positions[index + 1] - z,
                incoming_diameter_x=diameter_in_x,
                incoming_diameter_y=diameter_in_y,
                surface_footprint_x=surface_footprint_x,
                surface_footprint_y=surface_footprint_y,
                recommended_clear_x=1.5 * surface_footprint_x,
                recommended_clear_y=1.5 * surface_footprint_y,
            )
        )

    return StripeDesign(
        config=config,
        x_positive_focal=x_positive_focal,
        y_positive_focal=y_positive_focal,
        focus_distance=focus_distance,
        surface_z=surface_z,
        beam_at_surface=beam,
        surface_x_negative=x_negative,
        surface_x_positive=x_positive,
        surface_diameter_x=diameter_x,
        surface_diameter_y=diameter_y,
        prescription=prescription,
    )


def write_prescription_csv(design: StripeDesign, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "index",
                "name",
                "type",
                "z_mm",
                "focal_x_mm",
                "focal_y_mm",
                "distance_to_next_mm",
                "incoming_1e2_diameter_x_mm",
                "incoming_1e2_diameter_y_mm",
                "element_surface_1e2_footprint_x_mm",
                "element_surface_1e2_footprint_y_mm",
                "recommended_clear_x_mm",
                "recommended_clear_y_mm",
            ]
        )
        for lens in design.prescription:
            writer.writerow(
                [
                    lens.index,
                    lens.name,
                    lens.lens_type,
                    "{:.6f}".format(lens.z * 1e3),
                    "" if lens.focal_x is None else "{:.6f}".format(lens.focal_x * 1e3),
                    "" if lens.focal_y is None else "{:.6f}".format(lens.focal_y * 1e3),
                    "{:.6f}".format(lens.distance_to_next * 1e3),
                    "{:.6f}".format(lens.incoming_diameter_x * 1e3),
                    "{:.6f}".format(lens.incoming_diameter_y * 1e3),
                    "{:.6f}".format(lens.surface_footprint_x * 1e3),
                    "{:.6f}".format(lens.surface_footprint_y * 1e3),
                    "{:.6f}".format(lens.recommended_clear_x * 1e3),
                    "{:.6f}".format(lens.recommended_clear_y * 1e3),
                ]
            )
        writer.writerow(
            [
                len(design.prescription) + 1,
                "TARGET",
                "Plane normal at {:.3f} deg to beam".format(
                    design.config.incidence_angle_to_normal_deg
                ),
                "{:.6f}".format(design.surface_z * 1e3),
                "",
                "",
                "0.000000",
                "{:.9f}".format(2.0 * design.beam_at_surface.radius_x * 1e3),
                "{:.6f}".format(design.surface_diameter_y * 1e3),
                "{:.9f}".format(design.surface_diameter_x * 1e3),
                "{:.6f}".format(design.surface_diameter_y * 1e3),
                "",
                "",
            ]
        )
