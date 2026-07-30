"""X/Y 两方向独立的相干像散高斯光束。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import Optional

import numpy as np

from .gaussian import SPEED_OF_LIGHT, VACUUM_PERMITTIVITY


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError("{} 必须是正的有限数值。".format(name))


def _q_from_plane(wavelength: float, refractive_index: float, radius: float, curvature: float) -> complex:
    if curvature == 0 or np.isnan(curvature):
        raise ValueError("波前曲率半径不能为零或 NaN。")
    inverse_curvature = 0.0 if np.isinf(curvature) else 1.0 / curvature
    inverse_q = complex(
        inverse_curvature,
        -wavelength / (pi * refractive_index * radius ** 2),
    )
    return 1.0 / inverse_q


def beam_radius_from_q(q_value: complex, wavelength: float, refractive_index: float = 1.0) -> float:
    radius_squared = -wavelength / (pi * refractive_index * (1.0 / q_value).imag)
    return sqrt(radius_squared)


@dataclass(frozen=True)
class AstigmaticGaussianBeam:
    """某参考面处具有独立 qx、qy 的 TEM00 相干光束。"""

    wavelength: float
    refractive_index: float
    qx: complex
    qy: complex
    axis_field: complex
    power: float

    def __post_init__(self) -> None:
        _require_positive("真空波长", self.wavelength)
        _require_positive("折射率", self.refractive_index)
        _require_positive("功率", self.power)
        for name, q_value in (("qx", self.qx), ("qy", self.qy)):
            if not (isfinite(q_value.real) and isfinite(q_value.imag)) or q_value.imag <= 0:
                raise ValueError("{} 的虚部必须为正，且实部和虚部必须有限。".format(name))
        if not (isfinite(self.axis_field.real) and isfinite(self.axis_field.imag)):
            raise ValueError("轴上复电场必须有限。")

    @classmethod
    def from_plane(
        cls,
        wavelength: float,
        radius_x: float,
        radius_y: float,
        curvature_x: float = float("inf"),
        curvature_y: float = float("inf"),
        power: float = 1.0,
        refractive_index: float = 1.0,
        phase: float = 0.0,
    ) -> AstigmaticGaussianBeam:
        """由某平面的两个 1/e^2 半径及波前曲率建立光束。"""

        _require_positive("X 光斑半径", radius_x)
        _require_positive("Y 光斑半径", radius_y)
        _require_positive("真空波长", wavelength)
        _require_positive("折射率", refractive_index)
        _require_positive("功率", power)
        if not isfinite(phase):
            raise ValueError("初始相位必须有限。")

        qx = _q_from_plane(wavelength, refractive_index, radius_x, curvature_x)
        qy = _q_from_plane(wavelength, refractive_index, radius_y, curvature_y)
        field_magnitude = sqrt(
            4.0 * power
            / (pi * radius_x * radius_y * refractive_index * VACUUM_PERMITTIVITY * SPEED_OF_LIGHT)
        )
        return cls(
            wavelength=wavelength,
            refractive_index=refractive_index,
            qx=qx,
            qy=qy,
            axis_field=complex(field_magnitude * np.exp(1j * phase)),
            power=power,
        )

    @property
    def wave_number(self) -> float:
        return 2.0 * pi * self.refractive_index / self.wavelength

    @property
    def radius_x(self) -> float:
        return beam_radius_from_q(self.qx, self.wavelength, self.refractive_index)

    @property
    def radius_y(self) -> float:
        return beam_radius_from_q(self.qy, self.wavelength, self.refractive_index)

    @property
    def waist_radius_x(self) -> float:
        return sqrt(self.wavelength * self.qx.imag / (pi * self.refractive_index))

    @property
    def waist_radius_y(self) -> float:
        return sqrt(self.wavelength * self.qy.imag / (pi * self.refractive_index))

    @property
    def peak_intensity(self) -> float:
        return 0.5 * self.refractive_index * VACUUM_PERMITTIVITY * SPEED_OF_LIGHT * abs(
            self.axis_field
        ) ** 2

    def field(self, x, y):
        """计算当前垂直于光轴平面中的复电场 E(x,y)。"""

        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        phase = -1j * self.wave_number * (
            x_array ** 2 / (2.0 * self.qx) + y_array ** 2 / (2.0 * self.qy)
        )
        return self.axis_field * np.exp(phase)

    def intensity(self, x, y):
        electric_field = self.field(x, y)
        return (
            0.5
            * self.refractive_index
            * VACUUM_PERMITTIVITY
            * SPEED_OF_LIGHT
            * np.abs(electric_field) ** 2
        )

    def propagate(self, matrix_x, matrix_y, propagation_distance: float = 0.0) -> AstigmaticGaussianBeam:
        """分别使用 X/Y 的 ABCD 矩阵传播光束。"""

        if not isfinite(propagation_distance):
            raise ValueError("传播距离必须有限。")
        mx = np.asarray(matrix_x, dtype=np.float64)
        my = np.asarray(matrix_y, dtype=np.float64)
        if mx.shape != (2, 2) or my.shape != (2, 2):
            raise ValueError("X/Y ABCD 矩阵必须都是 2x2。")
        if not np.all(np.isfinite(mx)) or not np.all(np.isfinite(my)):
            raise ValueError("ABCD 矩阵必须有限。")

        def transform(q_value, matrix):
            a, b = float(matrix[0, 0]), float(matrix[0, 1])
            c, d = float(matrix[1, 0]), float(matrix[1, 1])
            return (a * q_value + b) / (c * q_value + d), a + b / q_value

        output_qx, field_denominator_x = transform(self.qx, mx)
        output_qy, field_denominator_y = transform(self.qy, my)
        field_denominator = np.sqrt(field_denominator_x * field_denominator_y)
        carrier_phase = np.exp(1j * self.wave_number * propagation_distance)
        output_field = self.axis_field * carrier_phase / field_denominator
        return AstigmaticGaussianBeam(
            wavelength=self.wavelength,
            refractive_index=self.refractive_index,
            qx=complex(output_qx),
            qy=complex(output_qy),
            axis_field=complex(output_field),
            power=self.power,
        )

    def free_space(self, distance: float) -> AstigmaticGaussianBeam:
        if not isfinite(distance):
            raise ValueError("传播距离必须有限。")
        matrix = np.array([[1.0, distance], [0.0, 1.0]], dtype=np.float64)
        return self.propagate(matrix, matrix, propagation_distance=distance)

    def apply_lens(
        self,
        focal_x: Optional[float] = None,
        focal_y: Optional[float] = None,
    ) -> AstigmaticGaussianBeam:
        """施加球面镜或只对一个方向有光焦度的柱面镜。"""

        def lens_matrix(focal_length):
            if focal_length is None or np.isinf(focal_length):
                return np.eye(2, dtype=np.float64)
            if not isfinite(focal_length) or focal_length == 0:
                raise ValueError("透镜焦距必须为非零有限数值。")
            return np.array([[1.0, 0.0], [-1.0 / focal_length, 1.0]], dtype=np.float64)

        return self.propagate(lens_matrix(focal_x), lens_matrix(focal_y))

