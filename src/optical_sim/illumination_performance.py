"""照明均匀度与效率的工程预算模型。

均匀度按线照明长轴 Y 的积分线强度定义；效率按激光器输出参考面到
晶圆目标矩形内的有用功率定义。该模型用于规格分配和验收判定，不能
替代基于实测镀膜曲线、DOE/匀化器衍射效率和真实像差的物理光学分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, isfinite, log, sqrt
from typing import Tuple

import numpy as np

from .dual_branch import BranchRequirement


@dataclass(frozen=True)
class IlluminationPerformanceRequirement:
    """四工况共同使用的性能门槛和建模约定。"""

    minimum_uniformity_ratio: float = 0.90
    minimum_useful_efficiency_ratio: float = 0.65
    uniformity_roi_y_fraction: float = 0.90
    design_supergaussian_order_y: float = 36.0
    uniformity_absolute_reserve: float = 0.04

    def __post_init__(self) -> None:
        for name, value in (
            ("最低均匀度", self.minimum_uniformity_ratio),
            ("最低效率", self.minimum_useful_efficiency_ratio),
            ("Y 向均匀度 ROI 比例", self.uniformity_roi_y_fraction),
        ):
            if not isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError("{} 必须位于 (0, 1]。".format(name))
        if not isfinite(self.design_supergaussian_order_y) or self.design_supergaussian_order_y <= 0.0:
            raise ValueError("超高斯阶数必须为正的有限数值。")
        if not isfinite(self.uniformity_absolute_reserve) or not 0.0 <= self.uniformity_absolute_reserve < 1.0:
            raise ValueError("均匀度绝对预留必须位于 [0, 1)。")


@dataclass(frozen=True)
class OpticalElementBudget:
    """一个元件或功能组的透过/反射效率分配。"""

    group: str
    element: str
    nominal_ratio: float
    allocated_minimum_ratio: float
    note: str = ""

    def __post_init__(self) -> None:
        if not self.group.strip() or not self.element.strip():
            raise ValueError("效率预算的组名和元件名不能为空。")
        for name, value in (
            ("标称效率", self.nominal_ratio),
            ("分配下限", self.allocated_minimum_ratio),
        ):
            if not isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError("{} 必须位于 (0, 1]。".format(name))
        if self.allocated_minimum_ratio > self.nominal_ratio:
            raise ValueError("分配下限不能高于标称效率。")


@dataclass(frozen=True)
class CasePerformanceResult:
    case_name: str
    uniformity_model_ratio: float
    uniformity_allocated_minimum_ratio: float
    path_efficiency_nominal_ratio: float
    path_efficiency_allocated_minimum_ratio: float
    target_capture_ratio: float
    useful_efficiency_nominal_ratio: float
    useful_efficiency_allocated_minimum_ratio: float
    uniformity_pass: bool
    efficiency_pass: bool


def supergaussian_edge_uniformity(order: float, roi_fraction: float) -> float:
    """返回中央 ROI 内的 I_min/I_max。

    使用 I(y)=exp[-2(|y|/w)^order]，其中 2w 是标称 1/e2 全宽，
    roi_fraction 是参与均匀度评价的标称全宽比例。
    """

    if not isfinite(order) or order <= 0.0:
        raise ValueError("超高斯阶数必须为正的有限数值。")
    if not isfinite(roi_fraction) or not 0.0 < roi_fraction <= 1.0:
        raise ValueError("ROI 比例必须位于 (0, 1]。")
    return exp(-2.0 * roi_fraction ** order)


def minimum_supergaussian_order(target_uniformity: float, roi_fraction: float) -> float:
    """求满足指定中央 ROI 均匀度所需的最小超高斯阶数。"""

    if not 0.0 < target_uniformity < 1.0:
        raise ValueError("目标均匀度必须位于 (0, 1)。")
    if not 0.0 < roi_fraction < 1.0:
        raise ValueError("ROI 比例必须位于 (0, 1)。")
    return log(-0.5 * log(target_uniformity)) / log(roi_fraction)


def supergaussian_capture_fraction(order: float) -> float:
    """数值计算超高斯在标称 1/e2 全宽内的功率比例。"""

    if not isfinite(order) or order <= 0.0:
        raise ValueError("超高斯阶数必须为正的有限数值。")
    coordinate = np.linspace(-4.0, 4.0, 20001)
    intensity = np.exp(-2.0 * np.abs(coordinate) ** order)
    inside = np.abs(coordinate) <= 1.0
    total = float(np.trapz(intensity, coordinate))
    captured = float(np.trapz(intensity[inside], coordinate[inside]))
    return captured / total


def default_efficiency_budget(requirement: BranchRequirement) -> Tuple[OpticalElementBudget, ...]:
    """返回与工况倍率、支路状态对应的 355 nm 效率预算。"""

    ratio_a = requirement.name.endswith("ratio_a")
    oblique = requirement.incidence_angle_to_normal_deg > 45.0
    common = (
        OpticalElementBudget("COMMON", "M0", 0.995, 0.993, "355 nm HR"),
        OpticalElementBudget("COMMON", "Attenuator open state", 0.985, 0.975, "验收时置最大透过"),
        OpticalElementBudget("COMMON", "Set Polarizer", 0.985, 0.975, "含波片/偏振损耗"),
        OpticalElementBudget("COMMON", "M2", 0.995, 0.993, "355 nm HR"),
        OpticalElementBudget("COMMON", "Servo samplers D1/D2", 0.990, 0.985, "两路取样合计"),
        OpticalElementBudget("COMMON", "Illumination Polarizer", 0.985, 0.975, "工作偏振状态"),
        OpticalElementBudget("COMMON", "Swathing Shutter", 0.995, 0.993, "全开状态"),
        OpticalElementBudget("COMMON", "Waist Relay", 0.990, 0.985, "多面 355 nm AR"),
        OpticalElementBudget("COMMON", "Y Homogenizer", 0.970, 0.950, "一维匀化/整形模块"),
        OpticalElementBudget("COMMON", "M3", 0.995, 0.993, "355 nm HR"),
        OpticalElementBudget("COMMON", "M4", 0.995, 0.993, "355 nm HR"),
        OpticalElementBudget(
            "MAGNIFICATION_SWITCH",
            "FIBE ratio {}".format("A" if ratio_a else "B"),
            0.985 if ratio_a else 0.980,
            0.980 if ratio_a else 0.970,
            "倍率档位相关",
        ),
    )
    if oblique:
        branch = (
            OpticalElementBudget("OBLIQUE", "Prism Diverter", 0.975, 0.970, "斜入射偏转状态"),
            OpticalElementBudget("OBLIQUE", "M5O", 0.995, 0.993, "355 nm HR"),
            OpticalElementBudget("OBLIQUE", "OBE", 0.975, 0.970, "斜支路扩束/整形"),
            OpticalElementBudget("OBLIQUE", "M6O", 0.995, 0.993, "355 nm HR"),
            OpticalElementBudget("OBLIQUE", "M7O", 0.995, 0.993, "355 nm HR"),
            OpticalElementBudget("OBLIQUE", "Oblique LFC", 0.990, 0.985, "末级柱面聚焦"),
        )
    else:
        branch = (
            OpticalElementBudget("NEAR_NORMAL", "Prism pass state", 0.990, 0.985, "近法向通过状态"),
            OpticalElementBudget("NEAR_NORMAL", "Fixed Expander", 0.980, 0.975, "近法向扩束"),
            OpticalElementBudget("NEAR_NORMAL", "M5N", 0.995, 0.993, "355 nm HR"),
            OpticalElementBudget("NEAR_NORMAL", "Normal LFC", 0.990, 0.985, "线形成柱面镜"),
            OpticalElementBudget("NEAR_NORMAL", "M6N", 0.995, 0.993, "355 nm HR"),
            OpticalElementBudget("NEAR_NORMAL", "AF beamsplitter region", 0.980, 0.975, "若实际不经过则删除"),
            OpticalElementBudget("NEAR_NORMAL", "Objective", 0.975, 0.970, "355 nm 透过率分配"),
        )
    return common + branch


def _product(values) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def evaluate_case_performance(
    requirement: BranchRequirement,
    performance: IlluminationPerformanceRequirement = IlluminationPerformanceRequirement(),
) -> CasePerformanceResult:
    """计算一个工况的均匀度和有用效率预算。"""

    budget = default_efficiency_budget(requirement)
    path_nominal = _product(item.nominal_ratio for item in budget)
    path_minimum = _product(item.allocated_minimum_ratio for item in budget)
    uniformity_model = supergaussian_edge_uniformity(
        performance.design_supergaussian_order_y,
        performance.uniformity_roi_y_fraction,
    )
    uniformity_minimum = max(0.0, uniformity_model - performance.uniformity_absolute_reserve)
    gaussian_x_capture = erf(sqrt(2.0))
    uniform_y_capture = supergaussian_capture_fraction(performance.design_supergaussian_order_y)
    capture = gaussian_x_capture * uniform_y_capture
    useful_nominal = path_nominal * capture
    useful_minimum = path_minimum * capture
    return CasePerformanceResult(
        case_name=requirement.name,
        uniformity_model_ratio=uniformity_model,
        uniformity_allocated_minimum_ratio=uniformity_minimum,
        path_efficiency_nominal_ratio=path_nominal,
        path_efficiency_allocated_minimum_ratio=path_minimum,
        target_capture_ratio=capture,
        useful_efficiency_nominal_ratio=useful_nominal,
        useful_efficiency_allocated_minimum_ratio=useful_minimum,
        uniformity_pass=uniformity_minimum >= performance.minimum_uniformity_ratio,
        efficiency_pass=useful_minimum >= performance.minimum_useful_efficiency_ratio,
    )
