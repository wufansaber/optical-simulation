"""可交互任意方向像散高斯光束传播模型。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, radians, sqrt, tan
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .astigmatic import AstigmaticGaussianBeam


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError("{} 必须是正的有限数值。".format(name))


@dataclass(frozen=True)
class PropagationDirection:
    """用 X-Z 与 Y-Z 平面内的方向角定义中心光线。"""

    theta_x_deg: float = 0.0
    theta_y_deg: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("X 方向角", self.theta_x_deg), ("Y 方向角", self.theta_y_deg)):
            if not isfinite(value) or abs(value) >= 89.0:
                raise ValueError("{} 必须位于 (-89, 89) deg。".format(name))

    @property
    def slope_x(self) -> float:
        return tan(radians(self.theta_x_deg))

    @property
    def slope_y(self) -> float:
        return tan(radians(self.theta_y_deg))

    @property
    def path_scale(self) -> float:
        return sqrt(1.0 + self.slope_x ** 2 + self.slope_y ** 2)

    @property
    def vector(self) -> np.ndarray:
        direction = np.array([self.slope_x, self.slope_y, 1.0], dtype=np.float64)
        return direction / np.linalg.norm(direction)

    @property
    def transverse_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        direction = self.vector
        basis_x = np.array([1.0, 0.0, -self.slope_x], dtype=np.float64)
        basis_x /= np.linalg.norm(basis_x)
        basis_y = np.cross(direction, basis_x)
        basis_y /= np.linalg.norm(basis_y)
        return basis_x, basis_y

    def path_length(self, delta_z: float) -> float:
        if not isfinite(delta_z):
            raise ValueError("Z 方向距离必须有限。")
        return delta_z * self.path_scale

    def center_at_z(self, z_m: float) -> Tuple[float, float]:
        return self.slope_x * z_m, self.slope_y * z_m


@dataclass(frozen=True)
class InteractiveLens:
    name: str
    z_m: float
    focal_x_m: Optional[float] = None
    focal_y_m: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("透镜名称不能为空。")
        if not isfinite(self.z_m) or self.z_m < 0.0:
            raise ValueError("透镜 Z 位置必须是非负有限数值。")
        for name, focal in (("X 焦距", self.focal_x_m), ("Y 焦距", self.focal_y_m)):
            if focal is not None and (not isfinite(focal) or focal == 0.0):
                raise ValueError("{} 必须为空或非零有限数值。".format(name))


@dataclass(frozen=True)
class InteractiveConfig:
    wavelength_m: float = 355e-9
    input_diameter_x_m: float = 1.2e-3
    input_diameter_y_m: float = 1.2e-3
    input_divergence_x_mrad: float = 0.0
    input_divergence_y_mrad: float = 0.0
    power_w: float = 1.0
    refractive_index: float = 1.0
    direction: PropagationDirection = PropagationDirection()

    def __post_init__(self) -> None:
        _positive("波长", self.wavelength_m)
        _positive("输入 X 光斑直径", self.input_diameter_x_m)
        _positive("输入 Y 光斑直径", self.input_diameter_y_m)
        _positive("功率", self.power_w)
        _positive("折射率", self.refractive_index)
        limit_mrad = 0.5 * pi * 1e3
        for name, value in (
            ("输入 X 发散半角", self.input_divergence_x_mrad),
            ("输入 Y 发散半角", self.input_divergence_y_mrad),
        ):
            if not isfinite(value) or abs(value) >= limit_mrad:
                raise ValueError("{} 必须是绝对值小于 pi/2 rad 的有限数值。".format(name))

    @staticmethod
    def _curvature_from_divergence(radius_m: float, divergence_mrad: float) -> float:
        """由 1/e2 半径处的局部包络斜率建立波前曲率。"""

        angle_rad = divergence_mrad * 1e-3
        if angle_rad == 0.0:
            return float("inf")
        return radius_m / tan(angle_rad)

    def input_beam(self) -> AstigmaticGaussianBeam:
        radius_x_m = 0.5 * self.input_diameter_x_m
        radius_y_m = 0.5 * self.input_diameter_y_m
        return AstigmaticGaussianBeam.from_plane(
            wavelength=self.wavelength_m,
            radius_x=radius_x_m,
            radius_y=radius_y_m,
            curvature_x=self._curvature_from_divergence(
                radius_x_m, self.input_divergence_x_mrad
            ),
            curvature_y=self._curvature_from_divergence(
                radius_y_m, self.input_divergence_y_mrad
            ),
            power=self.power_w,
            refractive_index=self.refractive_index,
        )


@dataclass(frozen=True)
class InteractiveBeamState:
    name: str
    z_m: float
    path_length_m: float
    center_x_m: float
    center_y_m: float
    beam: AstigmaticGaussianBeam


@dataclass(frozen=True)
class InteractiveResult:
    config: InteractiveConfig
    observation_z_m: float
    states: Tuple[InteractiveBeamState, ...]
    applied_lenses: Tuple[InteractiveLens, ...]
    ignored_lenses: Tuple[InteractiveLens, ...]

    @property
    def observation(self) -> InteractiveBeamState:
        return self.states[-1]


@dataclass(frozen=True)
class InteractivePathSample:
    z_m: float
    path_length_m: float
    radius_x_m: float
    radius_y_m: float
    center_x_m: float
    center_y_m: float


@dataclass(frozen=True)
class OutputPlaneIntensity:
    x_offset_m: np.ndarray
    y_offset_m: np.ndarray
    intensity_w_m2: np.ndarray
    normalized_intensity: np.ndarray
    center_x_m: float
    center_y_m: float


def _ordered_lenses(lenses: Sequence[InteractiveLens]) -> Tuple[InteractiveLens, ...]:
    return tuple(sorted(lenses, key=lambda lens: (lens.z_m, lens.name)))


def simulate_to_z(
    config: InteractiveConfig,
    lenses: Sequence[InteractiveLens],
    observation_z_m: float,
) -> InteractiveResult:
    """传播到任意全局 Z 平面；仅使用该平面之前的透镜。"""

    if not isfinite(observation_z_m) or observation_z_m < 0.0:
        raise ValueError("输出 Z 位置必须是非负有限数值。")
    ordered = _ordered_lenses(lenses)
    applied = tuple(lens for lens in ordered if lens.z_m <= observation_z_m)
    ignored = tuple(lens for lens in ordered if lens.z_m > observation_z_m)
    beam = config.input_beam()
    path_length = 0.0
    current_z = 0.0
    center_x, center_y = config.direction.center_at_z(0.0)
    states: List[InteractiveBeamState] = [
        InteractiveBeamState("LASER", 0.0, 0.0, center_x, center_y, beam)
    ]
    for lens in applied:
        delta_z = lens.z_m - current_z
        distance = config.direction.path_length(delta_z)
        beam = beam.free_space(distance).apply_lens(lens.focal_x_m, lens.focal_y_m)
        path_length += distance
        current_z = lens.z_m
        center_x, center_y = config.direction.center_at_z(current_z)
        states.append(
            InteractiveBeamState(lens.name, current_z, path_length, center_x, center_y, beam)
        )
    distance = config.direction.path_length(observation_z_m - current_z)
    beam = beam.free_space(distance)
    path_length += distance
    center_x, center_y = config.direction.center_at_z(observation_z_m)
    states.append(
        InteractiveBeamState("OUTPUT", observation_z_m, path_length, center_x, center_y, beam)
    )
    return InteractiveResult(config, observation_z_m, tuple(states), applied, ignored)


def sample_interactive_path(
    config: InteractiveConfig,
    lenses: Sequence[InteractiveLens],
    observation_z_m: float,
    samples_per_segment: int = 80,
) -> List[InteractivePathSample]:
    """采样从激光输出面到指定 Z 输出面的束半径与中心光线。"""

    if samples_per_segment < 2:
        raise ValueError("每段至少需要 2 个采样点。")
    result = simulate_to_z(config, lenses, observation_z_m)
    beam = config.input_beam()
    current_z = 0.0
    current_path = 0.0
    center_x, center_y = config.direction.center_at_z(0.0)
    samples = [
        InteractivePathSample(
            0.0,
            0.0,
            beam.radius_x,
            beam.radius_y,
            center_x,
            center_y,
        )
    ]
    endpoints: List[Tuple[float, Optional[InteractiveLens]]] = [
        (lens.z_m, lens) for lens in result.applied_lenses
    ]
    endpoints.append((observation_z_m, None))
    for endpoint_z, lens in endpoints:
        delta_z = endpoint_z - current_z
        if delta_z > 0.0:
            for fraction in np.linspace(0.0, 1.0, samples_per_segment + 1)[1:]:
                sample_z = current_z + fraction * delta_z
                distance = config.direction.path_length(sample_z - current_z)
                sample_beam = beam.free_space(distance)
                center_x, center_y = config.direction.center_at_z(sample_z)
                samples.append(
                    InteractivePathSample(
                        sample_z,
                        current_path + distance,
                        sample_beam.radius_x,
                        sample_beam.radius_y,
                        center_x,
                        center_y,
                    )
                )
            segment_distance = config.direction.path_length(delta_z)
            beam = beam.free_space(segment_distance)
            current_path += segment_distance
            current_z = endpoint_z
        if lens is not None:
            beam = beam.apply_lens(lens.focal_x_m, lens.focal_y_m)
    return samples


def intensity_on_output_z_plane(
    result: InteractiveResult,
    grid_points: int = 401,
    span_radii: float = 2.5,
) -> OutputPlaneIntensity:
    """计算全局 Z=常数平面上的二维光强，支持任意中心光线方向。"""

    if grid_points < 101 or grid_points % 2 == 0:
        raise ValueError("二维光场网格点数必须是大于等于 101 的奇数。")
    _positive("显示半径倍数", span_radii)
    observation = result.observation
    beam = observation.beam
    direction = result.config.direction.vector
    basis_x, basis_y = result.config.direction.transverse_basis
    mapping = np.array(
        [
            [basis_x[0], basis_x[1]],
            [basis_y[0], basis_y[1]],
        ],
        dtype=np.float64,
    )
    inverse_mapping = np.linalg.inv(mapping)
    local_limit_x = span_radii * beam.radius_x
    local_limit_y = span_radii * beam.radius_y
    global_limit_x = (
        abs(inverse_mapping[0, 0]) * local_limit_x
        + abs(inverse_mapping[0, 1]) * local_limit_y
    )
    global_limit_y = (
        abs(inverse_mapping[1, 0]) * local_limit_x
        + abs(inverse_mapping[1, 1]) * local_limit_y
    )
    x_offset = np.linspace(-global_limit_x, global_limit_x, grid_points)
    y_offset = np.linspace(-global_limit_y, global_limit_y, grid_points)
    x_grid, y_grid = np.meshgrid(x_offset, y_offset)
    local_x = basis_x[0] * x_grid + basis_x[1] * y_grid
    local_y = basis_y[0] * x_grid + basis_y[1] * y_grid
    axial_offset = direction[0] * x_grid + direction[1] * y_grid

    qx = beam.qx + axial_offset
    qy = beam.qy + axial_offset
    radius_x = np.sqrt(
        -beam.wavelength
        / (pi * beam.refractive_index * np.imag(1.0 / qx))
    )
    radius_y = np.sqrt(
        -beam.wavelength
        / (pi * beam.refractive_index * np.imag(1.0 / qy))
    )
    peak = 2.0 * beam.power / (pi * radius_x * radius_y)
    intensity = peak * np.exp(
        -2.0 * (local_x / radius_x) ** 2
        -2.0 * (local_y / radius_y) ** 2
    )
    normalized = intensity / float(np.max(intensity))
    return OutputPlaneIntensity(
        x_offset_m=x_offset,
        y_offset_m=y_offset,
        intensity_w_m2=intensity,
        normalized_intensity=normalized,
        center_x_m=observation.center_x_m,
        center_y_m=observation.center_y_m,
    )


def stripe_355nm_preset() -> Tuple[InteractiveLens, ...]:
    """返回当前 355 nm、0.8 um x 1500 um 条形光斑的七透镜处方。"""

    millimeter = 1e-3
    return (
        InteractiveLens("X1", 0.0, -15.0 * millimeter, None),
        InteractiveLens("X2", 153.153102 * millimeter, 168.153102 * millimeter, None),
        InteractiveLens("Y1", 183.153102 * millimeter, None, -50.0 * millimeter),
        InteractiveLens("Y2", 192.502142 * millimeter, None, 59.349040 * millimeter),
        InteractiveLens("C1", 222.502142 * millimeter, -100.0 * millimeter, -100.0 * millimeter),
        InteractiveLens("C2", 227.502142 * millimeter, 105.0 * millimeter, 105.0 * millimeter),
        InteractiveLens("F", 247.502142 * millimeter, 25.0 * millimeter, None),
    )
