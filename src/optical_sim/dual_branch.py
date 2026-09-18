"""180 mm 双支路照明系统的等效近轴高斯光束初算。

该模块用于在真实镜组处方缺失时建立可复现的第一版模型。每条支路
使用两个具有独立 X/Y 焦距的等效薄透镜面，在给定总光路长度内匹配
晶圆面的目标光斑。等效焦距不能直接作为真实镜片采购或装调数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, radians, sqrt, tan
from typing import Iterable, Tuple

import numpy as np


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError("{} 必须是正的有限数值。".format(name))


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError("{} 必须是有限数值。".format(name))


@dataclass(frozen=True)
class Paladin355Input:
    """Paladin Advanced 355-8000 的初算输入。

    divergence_half_angle_rad 使用 275 urad，是把规格表的 <550 urad
    暂按全角解释后的保守半角。规格定义确认后应替换该值。
    """

    wavelength_m: float = 355e-9
    minimum_power_w: float = 8.0
    beam_diameter_m: float = 1.0e-3
    divergence_half_angle_rad: float = 275e-6
    m2_x: float = 1.2
    m2_y: float = 1.2
    refractive_index: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("波长", self.wavelength_m),
            ("最小功率", self.minimum_power_w),
            ("光斑直径", self.beam_diameter_m),
            ("X 方向 M2", self.m2_x),
            ("Y 方向 M2", self.m2_y),
            ("折射率", self.refractive_index),
        ):
            _positive(name, value)
        _finite("发散半角", self.divergence_half_angle_rad)
        if abs(self.divergence_half_angle_rad) >= 0.5 * pi:
            raise ValueError("发散半角的绝对值必须小于 pi/2。")

    @property
    def input_radius_m(self) -> float:
        return 0.5 * self.beam_diameter_m

    def effective_wavelength(self, axis: str) -> float:
        if axis == "x":
            return self.m2_x * self.wavelength_m
        if axis == "y":
            return self.m2_y * self.wavelength_m
        raise ValueError("axis 必须是 x 或 y。")


@dataclass(frozen=True)
class BranchRequirement:
    name: str
    incidence_angle_to_normal_deg: float
    target_surface_diameter_x_m: float
    target_surface_diameter_y_m: float
    total_path_m: float = 180e-3
    lens1_s_m: float = 110e-3
    lens2_s_m: float = 174e-3
    objective_s_m: float | None = None
    objective_focal_m: float | None = None
    project_x_to_surface: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("支路名称不能为空。")
        if not 0.0 <= self.incidence_angle_to_normal_deg < 90.0:
            raise ValueError("入射角必须位于 [0, 90) deg。")
        for name, value in (
            ("目标 X 光斑", self.target_surface_diameter_x_m),
            ("目标 Y 光斑", self.target_surface_diameter_y_m),
            ("总光路长度", self.total_path_m),
            ("等效透镜 1 位置", self.lens1_s_m),
            ("等效透镜 2 位置", self.lens2_s_m),
        ):
            _positive(name, value)
        if not 0.0 < self.lens1_s_m < self.lens2_s_m < self.total_path_m:
            raise ValueError("必须满足 0 < lens1_s < lens2_s < total_path。")
        if (self.objective_s_m is None) != (self.objective_focal_m is None):
            raise ValueError("物镜位置和焦距必须同时填写或同时为空。")
        if self.objective_s_m is not None:
            _positive("物镜位置", self.objective_s_m)
            _positive("物镜等效焦距", self.objective_focal_m)
            if not self.lens2_s_m < self.objective_s_m < self.total_path_m:
                raise ValueError("必须满足 lens2_s < objective_s < total_path。")

    @property
    def surface_projection_x(self) -> float:
        if not self.project_x_to_surface:
            return 1.0
        return 1.0 / cos(radians(self.incidence_angle_to_normal_deg))

    @property
    def target_normal_radius_x_m(self) -> float:
        """垂直于主光线截面中的 X 半径。"""

        return 0.5 * self.target_surface_diameter_x_m / self.surface_projection_x

    @property
    def target_normal_radius_y_m(self) -> float:
        return 0.5 * self.target_surface_diameter_y_m


@dataclass(frozen=True)
class EquivalentLensPair:
    lens1_s_m: float
    lens2_s_m: float
    focal1_x_m: float
    focal1_y_m: float
    focal2_x_m: float
    focal2_y_m: float


@dataclass(frozen=True)
class BranchSimulationResult:
    requirement: BranchRequirement
    laser: Paladin355Input
    lenses: EquivalentLensPair
    output_normal_diameter_x_m: float
    output_normal_diameter_y_m: float
    output_surface_diameter_x_m: float
    output_surface_diameter_y_m: float
    paraxial_slope_x: float
    paraxial_slope_y: float
    geometric_na_x: float
    geometric_na_y: float
    ideal_peak_irradiance_w_m2: float

    @property
    def paraxial_valid_x(self) -> bool:
        return self.paraxial_slope_x <= 0.3

    @property
    def paraxial_valid_y(self) -> bool:
        return self.paraxial_slope_y <= 0.3


@dataclass(frozen=True)
class EstimatedElementPosition:
    branch: str
    element: str
    s_m: float
    role: str

    def __post_init__(self) -> None:
        if not self.branch.strip() or not self.element.strip() or not self.role.strip():
            raise ValueError("估算布局的支路、元件和功能不能为空。")
        if not isfinite(self.s_m) or not 0.0 <= self.s_m <= 180e-3:
            raise ValueError("估算位置必须位于 0 至 180 mm。")


def _input_q(laser: Paladin355Input, axis: str) -> complex:
    radius = laser.input_radius_m
    divergence = laser.divergence_half_angle_rad
    curvature = float("inf") if divergence == 0.0 else radius / tan(divergence)
    inverse_curvature = 0.0 if np.isinf(curvature) else 1.0 / curvature
    effective_wavelength = laser.effective_wavelength(axis)
    inverse_q = complex(
        inverse_curvature,
        -effective_wavelength / (pi * laser.refractive_index * radius ** 2),
    )
    return 1.0 / inverse_q


def _free_space(distance_m: float) -> np.ndarray:
    return np.array([[1.0, distance_m], [0.0, 1.0]], dtype=np.float64)


def _thin_lens(focal_m: float) -> np.ndarray:
    if not isfinite(focal_m) or focal_m == 0.0:
        raise ValueError("等效焦距必须是非零有限数值。")
    return np.array([[1.0, 0.0], [-1.0 / focal_m, 1.0]], dtype=np.float64)


def _system_matrix(requirement: BranchRequirement, focal1_m: float, focal2_m: float) -> np.ndarray:
    d1 = requirement.lens1_s_m
    d2 = requirement.lens2_s_m - requirement.lens1_s_m
    if requirement.objective_s_m is not None:
        d3 = requirement.objective_s_m - requirement.lens2_s_m
        d4 = requirement.total_path_m - requirement.objective_s_m
        return (
            _free_space(d4)
            @ _thin_lens(requirement.objective_focal_m)
            @ _free_space(d3)
            @ _thin_lens(focal2_m)
            @ _free_space(d2)
            @ _thin_lens(focal1_m)
            @ _free_space(d1)
        )
    d3 = requirement.total_path_m - requirement.lens2_s_m
    return (
        _free_space(d3)
        @ _thin_lens(focal2_m)
        @ _free_space(d2)
        @ _thin_lens(focal1_m)
        @ _free_space(d1)
    )


def _transform_q(q_value: complex, matrix: np.ndarray) -> complex:
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    denominator = c * q_value + d
    if abs(denominator) == 0.0:
        raise ValueError("ABCD 变换产生奇异 q 参数。")
    return (a * q_value + b) / denominator


def _radius_from_q(q_value: complex, effective_wavelength_m: float, refractive_index: float) -> float:
    inverse_q = 1.0 / q_value
    radius_squared = -effective_wavelength_m / (pi * refractive_index * inverse_q.imag)
    if radius_squared <= 0.0 or not isfinite(radius_squared):
        raise ValueError("q 参数不能产生有效光斑半径。")
    return sqrt(radius_squared)


def _target_q(radius_m: float, laser: Paladin355Input, axis: str) -> complex:
    effective_wavelength = laser.effective_wavelength(axis)
    rayleigh_range = pi * laser.refractive_index * radius_m ** 2 / effective_wavelength
    return complex(0.0, rayleigh_range)


def _newton_solve_pair(
    q_input: complex,
    q_target: complex,
    requirement: BranchRequirement,
    seed: Tuple[float, float],
) -> Tuple[float, float, float] | None:
    x = np.array(seed, dtype=np.float64)
    tolerance = max(1e-14, abs(q_target) * 1e-9)
    for _ in range(120):
        try:
            q_output = _transform_q(q_input, _system_matrix(requirement, x[0], x[1]))
        except ValueError:
            return None
        residual_complex = q_output - q_target
        residual = np.array([residual_complex.real, residual_complex.imag], dtype=np.float64)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance:
            return float(x[0]), float(x[1]), residual_norm

        jacobian = np.zeros((2, 2), dtype=np.float64)
        for index in range(2):
            step = max(1e-9, abs(x[index]) * 1e-6)
            trial = x.copy()
            trial[index] += step
            try:
                q_trial = _transform_q(
                    q_input,
                    _system_matrix(requirement, trial[0], trial[1]),
                )
            except ValueError:
                return None
            trial_residual = np.array(
                [(q_trial - q_target).real, (q_trial - q_target).imag],
                dtype=np.float64,
            )
            jacobian[:, index] = (trial_residual - residual) / step

        try:
            update = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return None

        accepted = False
        for exponent in range(30):
            trial = x + update * (0.5 ** exponent)
            if np.min(np.abs(trial)) <= 1e-6 or np.max(np.abs(trial)) > 10.0:
                continue
            try:
                q_trial = _transform_q(
                    q_input,
                    _system_matrix(requirement, trial[0], trial[1]),
                )
            except ValueError:
                continue
            if abs(q_trial - q_target) < abs(residual_complex):
                x = trial
                accepted = True
                break
        if not accepted:
            return None
    return None


def _candidate_seeds(target_radius_m: float, input_radius_m: float) -> Iterable[Tuple[float, float]]:
    if target_radius_m < input_radius_m:
        seeds = (
            (-0.020, 0.018),
            (-0.020, 0.013),
            (-0.012, 0.013),
            (-0.005, 0.018),
            (-0.100, 0.020),
            (-0.500, 0.020),
            (0.050, 0.020),
        )
    else:
        seeds = (
            (-0.020, -0.030),
            (-0.070, -0.027),
            (0.012, -0.044),
            (-0.030, 0.160),
            (-0.080, 0.210),
            (-0.200, 0.300),
            (-0.500, 0.500),
            (0.100, -0.100),
        )
    return seeds


def _solve_axis(
    laser: Paladin355Input,
    requirement: BranchRequirement,
    axis: str,
    target_radius_m: float,
) -> Tuple[float, float, complex]:
    q_input = _input_q(laser, axis)
    q_target = _target_q(target_radius_m, laser, axis)
    candidates = []
    for seed in _candidate_seeds(target_radius_m, laser.input_radius_m):
        solution = _newton_solve_pair(q_input, q_target, requirement, seed)
        if solution is None:
            continue
        focal1, focal2, residual = solution
        score = abs(1.0 / focal1) + abs(1.0 / focal2)
        if requirement.objective_s_m is not None and axis == "y":
            if focal1 >= 0.0 or focal2 >= 0.0:
                score += 1e6
        elif focal1 >= 0.0 or focal2 <= 0.0:
            score += 1e6
        candidates.append((score, residual, focal1, focal2))
    if not candidates:
        raise RuntimeError("{} 支路 {} 方向没有找到等效双透镜解。".format(requirement.name, axis.upper()))
    _, _, focal1, focal2 = min(candidates)
    q_output = _transform_q(q_input, _system_matrix(requirement, focal1, focal2))
    return focal1, focal2, q_output


def solve_equivalent_branch(
    requirement: BranchRequirement,
    laser: Paladin355Input | None = None,
) -> BranchSimulationResult:
    """求解双等效透镜面，使晶圆面成为 X/Y 共同束腰。"""

    laser = laser or Paladin355Input()
    focal1_x, focal2_x, qx = _solve_axis(
        laser,
        requirement,
        "x",
        requirement.target_normal_radius_x_m,
    )
    focal1_y, focal2_y, qy = _solve_axis(
        laser,
        requirement,
        "y",
        requirement.target_normal_radius_y_m,
    )
    radius_x = _radius_from_q(qx, laser.effective_wavelength("x"), laser.refractive_index)
    radius_y = _radius_from_q(qy, laser.effective_wavelength("y"), laser.refractive_index)
    normal_diameter_x = 2.0 * radius_x
    normal_diameter_y = 2.0 * radius_y
    surface_diameter_x = normal_diameter_x * requirement.surface_projection_x
    surface_diameter_y = normal_diameter_y
    slope_x = laser.effective_wavelength("x") / (
        pi * laser.refractive_index * requirement.target_normal_radius_x_m
    )
    slope_y = laser.effective_wavelength("y") / (
        pi * laser.refractive_index * requirement.target_normal_radius_y_m
    )
    geometric_na_x = laser.refractive_index * slope_x / sqrt(1.0 + slope_x ** 2)
    geometric_na_y = laser.refractive_index * slope_y / sqrt(1.0 + slope_y ** 2)
    ideal_peak_irradiance = 2.0 * laser.minimum_power_w / (
        pi
        * (0.5 * surface_diameter_x)
        * (0.5 * surface_diameter_y)
    )
    return BranchSimulationResult(
        requirement=requirement,
        laser=laser,
        lenses=EquivalentLensPair(
            lens1_s_m=requirement.lens1_s_m,
            lens2_s_m=requirement.lens2_s_m,
            focal1_x_m=focal1_x,
            focal1_y_m=focal1_y,
            focal2_x_m=focal2_x,
            focal2_y_m=focal2_y,
        ),
        output_normal_diameter_x_m=normal_diameter_x,
        output_normal_diameter_y_m=normal_diameter_y,
        output_surface_diameter_x_m=surface_diameter_x,
        output_surface_diameter_y_m=surface_diameter_y,
        paraxial_slope_x=slope_x,
        paraxial_slope_y=slope_y,
        geometric_na_x=geometric_na_x,
        geometric_na_y=geometric_na_y,
        ideal_peak_irradiance_w_m2=ideal_peak_irradiance,
    )


def sample_branch_envelope(
    result: BranchSimulationResult,
    samples_per_segment: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回展开光程坐标和垂直于主光线的 X/Y 1/e2 半径。"""

    if samples_per_segment < 2:
        raise ValueError("每段至少需要两个采样点。")
    requirement = result.requirement
    laser = result.laser
    elements_x = [
        (requirement.lens1_s_m, result.lenses.focal1_x_m),
        (requirement.lens2_s_m, result.lenses.focal2_x_m),
    ]
    elements_y = [
        (requirement.lens1_s_m, result.lenses.focal1_y_m),
        (requirement.lens2_s_m, result.lenses.focal2_y_m),
    ]
    if requirement.objective_s_m is not None:
        elements_x.append((requirement.objective_s_m, requirement.objective_focal_m))
        elements_y.append((requirement.objective_s_m, requirement.objective_focal_m))
    qx = _input_q(laser, "x")
    qy = _input_q(laser, "y")
    samples_s = [0.0]
    samples_x = [_radius_from_q(qx, laser.effective_wavelength("x"), laser.refractive_index)]
    samples_y = [_radius_from_q(qy, laser.effective_wavelength("y"), laser.refractive_index)]

    current_s = 0.0
    for element_index in range(len(elements_x) + 1):
        stop = (
            elements_x[element_index][0]
            if element_index < len(elements_x)
            else requirement.total_path_m
        )
        start = current_s
        for distance in np.linspace(0.0, stop - start, samples_per_segment + 1)[1:]:
            qx_sample = qx + float(distance)
            qy_sample = qy + float(distance)
            samples_s.append(start + float(distance))
            samples_x.append(
                _radius_from_q(qx_sample, laser.effective_wavelength("x"), laser.refractive_index)
            )
            samples_y.append(
                _radius_from_q(qy_sample, laser.effective_wavelength("y"), laser.refractive_index)
            )
        qx += stop - start
        qy += stop - start
        current_s = stop
        if element_index < len(elements_x):
            qx = _transform_q(qx, _thin_lens(elements_x[element_index][1]))
            qy = _transform_q(qy, _thin_lens(elements_y[element_index][1]))
    return (
        np.asarray(samples_s, dtype=np.float64),
        np.asarray(samples_x, dtype=np.float64),
        np.asarray(samples_y, dtype=np.float64),
    )


def sample_paraxial_ray_bundle(
    result: BranchSimulationResult,
    axis: str,
    ray_count: int = 17,
    samples_per_segment: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """追迹与高斯 q 参数相匹配的 1/e2 相空间光线束。

    返回累计展开光程 ``s`` 和形状为 ``(len(s), ray_count)`` 的光线高度。
    周边光线从输入面的 1/e2 相空间椭圆采样，最后一条为中心主光线。
    这是等效薄透镜近轴追迹；不代表真实曲面或高 NA 矢量追迹。
    """

    if axis not in ("x", "y"):
        raise ValueError("axis 必须是 x 或 y。")
    if ray_count < 5:
        raise ValueError("光线数至少为 5。")
    if samples_per_segment < 2:
        raise ValueError("每段至少需要两个采样点。")

    requirement = result.requirement
    laser = result.laser
    radius = laser.input_radius_m
    divergence = laser.divergence_half_angle_rad
    inverse_curvature = 0.0 if divergence == 0.0 else tan(divergence) / radius
    diffraction_angle = (
        laser.effective_wavelength(axis)
        / (pi * laser.refractive_index * radius)
    )
    phase = np.linspace(0.0, 2.0 * pi, ray_count - 1, endpoint=False)
    heights = radius * np.cos(phase)
    angles = heights * inverse_curvature + diffraction_angle * np.sin(phase)
    heights = np.concatenate((heights, np.asarray([0.0])))
    angles = np.concatenate((angles, np.asarray([0.0])))

    elements = list(
        (
            (requirement.lens1_s_m, result.lenses.focal1_x_m),
            (requirement.lens2_s_m, result.lenses.focal2_x_m),
        )
        if axis == "x"
        else (
            (requirement.lens1_s_m, result.lenses.focal1_y_m),
            (requirement.lens2_s_m, result.lenses.focal2_y_m),
        )
    )
    if requirement.objective_s_m is not None:
        elements.append((requirement.objective_s_m, requirement.objective_focal_m))
    samples_s = [0.0]
    samples_rays = [heights.copy()]

    current_s = 0.0
    for element_index in range(len(elements) + 1):
        stop = elements[element_index][0] if element_index < len(elements) else requirement.total_path_m
        distance_total = stop - current_s
        for distance in np.linspace(0.0, distance_total, samples_per_segment + 1)[1:]:
            samples_s.append(current_s + float(distance))
            samples_rays.append(heights + float(distance) * angles)
        heights = heights + distance_total * angles
        current_s = stop
        if element_index < len(elements):
            angles = angles - heights / elements[element_index][1]

    return np.asarray(samples_s, dtype=np.float64), np.asarray(samples_rays, dtype=np.float64)


def default_dual_branch_requirements() -> Tuple[
    BranchRequirement,
    BranchRequirement,
    BranchRequirement,
    BranchRequirement,
]:
    """返回两种入射角乘以两个倍率档位的四个工况。"""

    return (
        BranchRequirement(
            name="near_normal_1deg_ratio_a",
            incidence_angle_to_normal_deg=1.0,
            target_surface_diameter_x_m=1.44e-6,
            target_surface_diameter_y_m=5.2e-3,
            lens2_s_m=166e-3,
            objective_s_m=174e-3,
            objective_focal_m=30e-3,
        ),
        BranchRequirement(
            name="near_normal_1deg_ratio_b",
            incidence_angle_to_normal_deg=1.0,
            target_surface_diameter_x_m=1.05e-6,
            target_surface_diameter_y_m=2.667e-3,
            lens2_s_m=166e-3,
            objective_s_m=174e-3,
            objective_focal_m=30e-3,
        ),
        BranchRequirement(
            name="oblique_75deg_ratio_a",
            incidence_angle_to_normal_deg=75.0,
            target_surface_diameter_x_m=1.44e-6,
            target_surface_diameter_y_m=5.2e-3,
        ),
        BranchRequirement(
            name="oblique_75deg_ratio_b",
            incidence_angle_to_normal_deg=75.0,
            target_surface_diameter_x_m=1.05e-6,
            target_surface_diameter_y_m=2.667e-3,
        ),
    )


def default_estimated_layout() -> Tuple[EstimatedElementPosition, ...]:
    """按系统图比例和 180 mm 总长给出第一版展开位置预算。

    s=0 定义为激光器输出参考面。激光器本体长度 90 mm 只用于机械
    缩放，不计入该输出面之后的 180 mm 展开传播距离。
    """

    return (
        EstimatedElementPosition("COMMON", "LASER_OUTPUT", 0.000, "激光器输出参考面"),
        EstimatedElementPosition("COMMON", "M0", 0.006, "第一稳光反射镜"),
        EstimatedElementPosition("COMMON", "ATTENUATOR", 0.012, "功率调节"),
        EstimatedElementPosition("COMMON", "SET_POLARIZER", 0.018, "初始偏振设置"),
        EstimatedElementPosition("COMMON", "M2", 0.026, "第二稳光反射镜"),
        EstimatedElementPosition("COMMON", "SERVO_PICKOFF", 0.034, "稳光分光与探测"),
        EstimatedElementPosition("COMMON", "ILLUMINATION_POLARIZER", 0.044, "工作偏振设置"),
        EstimatedElementPosition("COMMON", "SWATHING_SHUTTER", 0.052, "照明快门"),
        EstimatedElementPosition("COMMON", "WAIST_RELAY_1", 0.060, "腰斑中继第一面"),
        EstimatedElementPosition("COMMON", "WAIST_RELAY_2", 0.072, "腰斑中继第二面"),
        EstimatedElementPosition("COMMON", "HOMOGENIZER_Y", 0.075, "Y 向一维匀化，目标均匀度不低于 90%"),
        EstimatedElementPosition("COMMON", "M3", 0.078, "抬升折转"),
        EstimatedElementPosition("COMMON", "M4", 0.086, "路径对准"),
        EstimatedElementPosition("COMMON", "FIBE", 0.092, "倍率切换"),
        EstimatedElementPosition("COMMON", "PRISM_DIVERTER", 0.100, "正常与斜入射分支点"),
        EstimatedElementPosition("OBLIQUE", "M5O", 0.104, "OBE 前端对准"),
        EstimatedElementPosition("OBLIQUE", "OBE_EQUIVALENT_E1", 0.110, "倍率相关非对称扩束等效面"),
        EstimatedElementPosition("OBLIQUE", "M6O", 0.145, "OBE 后端对准"),
        EstimatedElementPosition("OBLIQUE", "M7O", 0.160, "设置 75 度入射和晶圆交点"),
        EstimatedElementPosition("OBLIQUE", "OBLIQUE_LFC_E2", 0.174, "短工作距线形成柱面镜与弱 Y 校正"),
        EstimatedElementPosition("OBLIQUE", "WAFER", 0.180, "晶圆面，工作距 6 mm"),
        EstimatedElementPosition("NEAR_NORMAL", "NEAR_NORMAL_EXPANDER_E1", 0.110, "倍率相关扩束等效面"),
        EstimatedElementPosition("NEAR_NORMAL", "M5N", 0.145, "近法向支路对准"),
        EstimatedElementPosition("NEAR_NORMAL", "NORMAL_LFC_E2", 0.166, "近法向线形成与弱 Y 校正等效面"),
        EstimatedElementPosition("NEAR_NORMAL", "M6N", 0.170, "折转到物镜照明轴"),
        EstimatedElementPosition("NEAR_NORMAL", "OBJECTIVE_IDEAL", 0.174, "30 mm 等效焦距理想薄透镜面"),
        EstimatedElementPosition("NEAR_NORMAL", "WAFER", 0.180, "晶圆面，工作距 6 mm"),
    )
