"""相干 TEM00 高斯光束的复 q 参数与 ABCD 矩阵传播。"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, pi, sqrt
from typing import List

import numpy as np

from .paraxial import FreeSpace, OpticalSystem, ThinLens

SPEED_OF_LIGHT = 299_792_458.0
VACUUM_PERMITTIVITY = 8.854_187_812_8e-12


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError("{} 必须是正的有限数值。".format(name))


@dataclass(frozen=True)
class GaussianBeam:
    """某一参考面处的相干高斯光束状态。

    wavelength 是真空波长，q 是该参考面上的复光束参数，axis_field
    是该平面光轴处的复电场相量，单位为 V/m。
    """

    wavelength: float
    refractive_index: float
    q: complex
    axis_field: complex
    power: float

    def __post_init__(self) -> None:
        _require_positive("真空波长", self.wavelength)
        _require_positive("折射率", self.refractive_index)
        _require_positive("功率", self.power)
        if not (isfinite(self.q.real) and isfinite(self.q.imag)) or self.q.imag <= 0:
            raise ValueError("复光束参数 q 的虚部必须为正，且实部和虚部必须有限。")
        if not (isfinite(self.axis_field.real) and isfinite(self.axis_field.imag)):
            raise ValueError("轴上复电场必须有限。")

    @classmethod
    def from_waist(
        cls,
        wavelength: float,
        waist_radius: float,
        waist_position: float = 0.0,
        reference_position: float = 0.0,
        power: float = 1.0,
        refractive_index: float = 1.0,
        phase: float = 0.0,
    ) -> GaussianBeam:
        """用已知束腰半径和全局束腰位置建立参考面上的光束。"""

        _require_positive("束腰半径", waist_radius)
        _require_positive("真空波长", wavelength)
        _require_positive("折射率", refractive_index)
        _require_positive("功率", power)
        if not all(isfinite(value) for value in (waist_position, reference_position, phase)):
            raise ValueError("位置和相位必须是有限数值。")

        rayleigh_range = pi * refractive_index * waist_radius ** 2 / wavelength
        q_value = complex(reference_position - waist_position, rayleigh_range)
        beam_radius = waist_radius * sqrt(
            1.0 + ((reference_position - waist_position) / rayleigh_range) ** 2
        )
        field_magnitude = sqrt(
            4.0 * power
            / (pi * beam_radius ** 2 * refractive_index * VACUUM_PERMITTIVITY * SPEED_OF_LIGHT)
        )
        axis_field = field_magnitude * np.exp(1j * phase)
        return cls(wavelength, refractive_index, q_value, complex(axis_field), power)

    @classmethod
    def from_plane(
        cls,
        wavelength: float,
        beam_radius: float,
        radius_of_curvature: float = float("inf"),
        power: float = 1.0,
        refractive_index: float = 1.0,
        phase: float = 0.0,
    ) -> GaussianBeam:
        """用某平面的 1/e^2 光斑半径和波前曲率建立光束。

        平面波前使用无穷大曲率。曲率符号遵循 1/q = 1/R - i lambda/(pi*n*w^2)。
        """

        _require_positive("光斑半径", beam_radius)
        _require_positive("真空波长", wavelength)
        _require_positive("折射率", refractive_index)
        _require_positive("功率", power)
        if radius_of_curvature == 0 or np.isnan(radius_of_curvature):
            raise ValueError("波前曲率半径不能为零或 NaN。")
        if not isfinite(phase):
            raise ValueError("相位必须有限。")

        inverse_curvature = 0.0 if np.isinf(radius_of_curvature) else 1.0 / radius_of_curvature
        inverse_q = complex(
            inverse_curvature,
            -wavelength / (pi * refractive_index * beam_radius ** 2),
        )
        q_value = 1.0 / inverse_q
        field_magnitude = sqrt(
            4.0 * power
            / (pi * beam_radius ** 2 * refractive_index * VACUUM_PERMITTIVITY * SPEED_OF_LIGHT)
        )
        axis_field = field_magnitude * np.exp(1j * phase)
        return cls(wavelength, refractive_index, q_value, complex(axis_field), power)

    @property
    def wave_number(self) -> float:
        return 2.0 * pi * self.refractive_index / self.wavelength

    @property
    def rayleigh_range(self) -> float:
        """与当前 q 对应的束腰瑞利长度。"""

        return self.q.imag

    @property
    def waist_radius(self) -> float:
        return sqrt(self.wavelength * self.q.imag / (pi * self.refractive_index))

    @property
    def beam_radius(self) -> float:
        inverse_q = 1.0 / self.q
        radius_squared = -self.wavelength / (
            pi * self.refractive_index * inverse_q.imag
        )
        return sqrt(radius_squared)

    @property
    def radius_of_curvature(self) -> float:
        inverse_curvature = (1.0 / self.q).real
        if abs(inverse_curvature) < 1e-15:
            return float("inf")
        return 1.0 / inverse_curvature

    @property
    def gouy_phase(self) -> float:
        return atan2(self.q.real, self.q.imag)

    @property
    def divergence_half_angle(self) -> float:
        return self.wavelength / (pi * self.refractive_index * self.waist_radius)

    @property
    def peak_intensity(self) -> float:
        """当前平面轴上峰值强度，单位 W/m^2。"""

        return 0.5 * self.refractive_index * VACUUM_PERMITTIVITY * SPEED_OF_LIGHT * abs(
            self.axis_field
        ) ** 2

    def waist_position(self, reference_position: float) -> float:
        """返回当前均匀介质区中由 q 预测的全局束腰位置。"""

        if not isfinite(reference_position):
            raise ValueError("参考面位置必须有限。")
        return reference_position - self.q.real

    def field(self, radius):
        """计算当前平面给定横向半径处的复电场相量，单位 V/m。"""

        radius_array = np.asarray(radius, dtype=np.float64)
        return self.axis_field * np.exp(-1j * self.wave_number * radius_array ** 2 / (2.0 * self.q))

    def intensity(self, radius):
        """计算当前平面的时间平均光强，单位 W/m^2。"""

        electric_field = self.field(radius)
        return (
            0.5
            * self.refractive_index
            * VACUUM_PERMITTIVITY
            * SPEED_OF_LIGHT
            * np.abs(electric_field) ** 2
        )

    def propagate(self, matrix, propagation_distance: float = 0.0) -> GaussianBeam:
        """使用实 ABCD 矩阵传播 q 和相干轴上复电场。"""

        if not isfinite(propagation_distance) or propagation_distance < 0:
            raise ValueError("传播距离必须是非负有限数值。")
        abcd = np.asarray(matrix, dtype=np.float64)
        if abcd.shape != (2, 2) or not np.all(np.isfinite(abcd)):
            raise ValueError("ABCD 矩阵必须是有限的 2x2 实矩阵。")

        a, b = float(abcd[0, 0]), float(abcd[0, 1])
        c, d = float(abcd[1, 0]), float(abcd[1, 1])
        q_denominator = c * self.q + d
        if abs(q_denominator) == 0:
            raise ValueError("ABCD 变换产生了奇异 q 参数。")
        output_q = (a * self.q + b) / q_denominator

        field_denominator = a + b / self.q
        carrier_phase = np.exp(1j * self.wave_number * propagation_distance)
        output_axis_field = self.axis_field * carrier_phase / field_denominator
        return GaussianBeam(
            wavelength=self.wavelength,
            refractive_index=self.refractive_index,
            q=complex(output_q),
            axis_field=complex(output_axis_field),
            power=self.power,
        )


@dataclass(frozen=True)
class GaussianTracePoint:
    z: float
    beam: GaussianBeam
    element: str


def sample_gaussian_system(
    system: OpticalSystem,
    input_beam: GaussianBeam,
    samples_per_space: int = 160,
) -> List[GaussianTracePoint]:
    """沿系统自由空间连续采样，并在每个薄透镜后更新 q。"""

    if samples_per_space < 1:
        raise ValueError("每段自由空间至少需要一个采样点。")

    current = input_beam
    z = 0.0
    points = [GaussianTracePoint(z, current, "input")]

    for element in system.elements:
        if isinstance(element, FreeSpace):
            start_beam = current
            start_z = z
            distances = np.linspace(0.0, element.distance, samples_per_space + 1)[1:]
            for distance in distances:
                sampled_beam = start_beam.propagate(
                    FreeSpace(float(distance)).matrix,
                    propagation_distance=float(distance),
                )
                points.append(
                    GaussianTracePoint(
                        start_z + float(distance),
                        sampled_beam,
                        "free-space",
                    )
                )
            current = points[-1].beam if element.distance > 0 else current
            z += element.distance
        elif isinstance(element, ThinLens):
            current = current.propagate(element.matrix)
            points.append(GaussianTracePoint(z, current, "thin-lens"))
        else:
            current = current.propagate(element.matrix)
            points.append(GaussianTracePoint(z, current, type(element).__name__))

    return points


def propagate_gaussian_system(
    system: OpticalSystem,
    input_beam: GaussianBeam,
) -> GaussianTracePoint:
    """只计算系统最终输出状态。"""

    return sample_gaussian_system(system, input_beam, samples_per_space=1)[-1]

