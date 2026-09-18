"""双支路照明系统的三维工程预布局。

本模块把 180 mm 展开光程映射为一套内部自洽的折叠三维坐标。坐标
用于方案评审、机械接口预留和后续非序列模型初始化，不是实测机械
坐标。晶圆中心和两种入射角是硬约束，其余折转位置属于工程预估。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, degrees, radians, sin, sqrt
from typing import Iterable, Tuple

import numpy as np


Vector3 = Tuple[float, float, float]


def _unit(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(vector), dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("方向向量不能为零。")
    return value / norm


def _tuple(vector: np.ndarray) -> Vector3:
    return tuple(float(value) for value in vector)  # type: ignore[return-value]


def yaw_pitch_deg(direction: Iterable[float]) -> Tuple[float, float]:
    """返回单位向量的方位角和俯仰角。"""

    x, y, z = _unit(direction)
    return degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y)))


def reflection_normal(incoming: Iterable[float], outgoing: Iterable[float]) -> np.ndarray:
    """返回满足镜面反射的一个法向；法向反号表示同一平面。"""

    return _unit(_unit(incoming) - _unit(outgoing))


def mirror_incidence_deg(incoming: Iterable[float], normal: Iterable[float]) -> float:
    cosine = abs(float(np.dot(_unit(incoming), _unit(normal))))
    return degrees(acos(max(-1.0, min(1.0, cosine))))


@dataclass(frozen=True)
class LayoutElement3D:
    element_id: str
    path_group: str
    component_type: str
    s_mm: float | None
    position_mm: Vector3
    chief_ray_direction: Vector3 | None
    surface_normal: Vector3 | None
    aperture_x_mm: float
    aperture_y_mm: float
    function: str
    switch_state: str = "固定"
    basis: str = "工程预估"
    note: str = ""


@dataclass(frozen=True)
class PathVertex3D:
    path_id: str
    sequence: int
    element_id: str
    s_mm: float
    position_mm: Vector3


@dataclass(frozen=True)
class Layout3D:
    elements: Tuple[LayoutElement3D, ...]
    near_path: Tuple[PathVertex3D, ...]
    oblique_path: Tuple[PathVertex3D, ...]
    near_direction: Vector3
    oblique_direction: Vector3
    wafer_normal: Vector3 = (0.0, 0.0, 1.0)


def _point_on(start: np.ndarray, direction: np.ndarray, distance_mm: float) -> np.ndarray:
    return start + distance_mm * direction


def _transmissive(
    element_id: str,
    group: str,
    component_type: str,
    s_mm: float,
    position: np.ndarray,
    direction: np.ndarray,
    aperture: Tuple[float, float],
    function: str,
    switch_state: str = "固定",
    basis: str = "工程预估",
    note: str = "",
) -> LayoutElement3D:
    return LayoutElement3D(
        element_id,
        group,
        component_type,
        s_mm,
        _tuple(position),
        _tuple(direction),
        _tuple(direction),
        aperture[0],
        aperture[1],
        function,
        switch_state,
        basis,
        note,
    )


def _mirror(
    element_id: str,
    group: str,
    s_mm: float,
    position: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    aperture: Tuple[float, float],
    function: str,
    basis: str = "工程预估",
) -> LayoutElement3D:
    normal = reflection_normal(incoming, outgoing)
    return LayoutElement3D(
        element_id,
        group,
        "平面反射镜",
        s_mm,
        _tuple(position),
        _tuple(incoming),
        _tuple(normal),
        aperture[0],
        aperture[1],
        function,
        "固定；二维微调架",
        basis,
        "标称入射角 {:.3f} deg".format(mirror_incidence_deg(incoming, normal)),
    )


def default_layout_3d() -> Layout3D:
    """返回一套满足两条 180 mm 光程和 1°/75°入射的三维预布局。"""

    ex = np.array((1.0, 0.0, 0.0))
    ey = np.array((0.0, 1.0, 0.0))
    ez = np.array((0.0, 0.0, 1.0))
    wafer = np.zeros(3)
    near_direction = _unit((sin(radians(1.0)), 0.0, -cos(radians(1.0))))
    oblique_direction = _unit((sin(radians(75.0)), 0.0, -cos(radians(75.0))))

    # 晶圆附近由工作距反推：物镜至晶圆 6 mm，M6N 至晶圆 10 mm。
    objective = wafer - 6.0 * near_direction
    m6n = wafer - 10.0 * near_direction
    fourier = wafer - 9.0 * near_direction
    af_beamsplitter = wafer - 8.0 * near_direction
    oblique_lfc = wafer - 6.0 * oblique_direction
    m7o = wafer - 20.0 * oblique_direction

    # 分支点选择在晶圆左后上方。近法向通过方向同时作为共用段末段方向。
    prism = np.array((-50.0, -30.0, 20.0))
    near_remaining = m6n - prism
    horizontal = near_remaining.copy()
    horizontal[2] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal))
    cosine_delta = (
        float(np.dot(near_remaining, near_remaining)) + 45.0 ** 2 - 25.0 ** 2
    ) / (2.0 * 45.0 * horizontal_norm)
    horizontal_azimuth = atan2(horizontal[1], horizontal[0])
    near_pass_direction = _unit(
        (
            cos(horizontal_azimuth - acos(cosine_delta)),
            sin(horizontal_azimuth - acos(cosine_delta)),
            0.0,
        )
    )
    near_fold_direction = _unit((near_remaining - 45.0 * near_pass_direction) / 25.0)

    # 公共段反向构造，保证每一段欧氏距离与累计光程 s 完全一致。
    m4 = prism - 14.0 * near_pass_direction
    m3 = m4 - 8.0 * ez
    m2 = m3 - 52.0 * ex
    m0 = m2 - 20.0 * ey
    laser_output = m0 - 6.0 * ex

    attenuator = _point_on(m0, ey, 6.0)
    set_polarizer = _point_on(m0, ey, 12.0)
    bs1 = _point_on(m2, ex, 6.0)
    bs2 = _point_on(m2, ex, 10.0)
    illumination_polarizer = _point_on(m2, ex, 18.0)
    shutter = _point_on(m2, ex, 26.0)
    waist_1 = _point_on(m2, ex, 34.0)
    waist_2 = _point_on(m2, ex, 46.0)
    homogenizer = _point_on(m2, ex, 49.0)
    fibe = _point_on(m4, near_pass_direction, 6.0)

    near_e1 = _point_on(prism, near_pass_direction, 10.0)
    m5n = _point_on(prism, near_pass_direction, 45.0)
    normal_lfc = _point_on(m5n, near_fold_direction, 21.0)

    # 斜入射支路：Prism 向 -Y 偏转，随后经 M5O/M6O/M7O 折到 75°主光线。
    prism_to_m5o = -ey
    m5o = _point_on(prism, prism_to_m5o, 4.0)
    remaining = m7o - m5o
    remaining_length = float(np.linalg.norm(remaining))
    cosine_alpha = (
        remaining_length ** 2 + 41.0 ** 2 - 15.0 ** 2
    ) / (2.0 * 41.0 * remaining_length)
    remaining_direction = remaining / remaining_length
    normal_to_remaining = ez - float(np.dot(ez, remaining_direction)) * remaining_direction
    normal_to_remaining = _unit(normal_to_remaining)
    alpha = acos(cosine_alpha)
    oblique_mid_direction = _unit(
        cosine_alpha * remaining_direction + sin(alpha) * normal_to_remaining
    )
    m6o = _point_on(m5o, oblique_mid_direction, 41.0)
    oblique_down_direction = _unit((m7o - m6o) / 15.0)
    obe_e1 = _point_on(m5o, oblique_mid_direction, 6.0)

    elements = []
    elements.extend(
        (
            LayoutElement3D(
                "LASER_BODY",
                "COMMON",
                "Paladin 355-8000 激光器",
                -45.0,
                _tuple(laser_output - 45.0 * ex),
                _tuple(ex),
                None,
                25.0,
                25.0,
                "355 nm、>8 W；机械长度按 90 mm 预留",
                "固定",
                "长度已确认；外形截面预估",
                "中心坐标；机身沿 X，输出端为 s=0",
            ),
            _transmissive(
                "INTERNAL_SAFETY_SHUTTER",
                "COMMON",
                "内置安全快门",
                -3.0,
                laser_output - 3.0 * ex,
                ex,
                (5.0, 5.0),
                "激光器内部安全联锁",
                basis="图示位置预估",
            ),
            _transmissive(
                "LASER_OUTPUT",
                "COMMON",
                "输出参考面",
                0.0,
                laser_output,
                ex,
                (3.0, 3.0),
                "180 mm 照明光程起点",
                basis="坐标基准定义",
                note="输入 1/e² 直径 1.0 mm，发散全角 <550 urad",
            ),
            _mirror("M0", "COMMON", 6.0, m0, ex, ey, (8.0, 8.0), "第一稳光伺服镜"),
            _transmissive("ATTENUATOR", "COMMON", "可变衰减器", 12.0, attenuator, ey, (8.0, 8.0), "照明功率设置", note="验收效率时置最大透过"),
            _transmissive("SET_POLARIZER", "COMMON", "1/4+1/2 波片组", 18.0, set_polarizer, ey, (8.0, 8.0), "C/S/P 偏振状态设置", switch_state="四种波片组合状态"),
            _mirror("M2", "COMMON", 26.0, m2, ey, ex, (8.0, 8.0), "第二稳光伺服镜"),
            _transmissive("SERVO_BS1", "COMMON", "稳光取样分光片", 32.0, bs1, ex, (8.0, 8.0), "Detector 1 取样", note="主光透过，少量向 +Z 取样"),
            _transmissive("SERVO_BS2", "COMMON", "稳光取样分光片", 36.0, bs2, ex, (8.0, 8.0), "Detector 2 取样", note="主光透过，少量向 -Y 取样"),
            LayoutElement3D("DETECTOR_1", "DIAGNOSTIC", "四象限探测器", None, _tuple(bs1 + 8.0 * ez), _tuple(ez), _tuple(-ez), 5.0, 5.0, "M0/M2 指向反馈通道 1", basis="离轴安装预估"),
            LayoutElement3D("DETECTOR_2", "DIAGNOSTIC", "四象限探测器", None, _tuple(bs2 - 8.0 * ey), _tuple(-ey), _tuple(ey), 5.0, 5.0, "M0/M2 指向反馈通道 2", basis="离轴安装预估"),
            _transmissive("ILLUMINATION_POLARIZER", "COMMON", "照明偏振器", 44.0, illumination_polarizer, ex, (8.0, 8.0), "工作偏振选择"),
            _transmissive("SWATHING_SHUTTER", "COMMON", "照明快门", 52.0, shutter, ex, (8.0, 8.0), "扫描/曝光门控"),
            _transmissive("WAIST_RELAY_1", "COMMON", "腰斑中继透镜 1", 60.0, waist_1, ex, (10.0, 10.0), "激光腰斑中继第一面", note="焦距待真实处方替换"),
            _transmissive("WAIST_RELAY_2", "COMMON", "腰斑中继透镜 2", 72.0, waist_2, ex, (10.0, 10.0), "设置中继腰斑直径", note="焦距待真实处方替换"),
            _transmissive("HOMOGENIZER_Y", "COMMON", "一维 Y 向匀化器", 75.0, homogenizer, ex, (10.0, 10.0), "形成 36 阶超高斯长轴，中央 90% ROI 均匀度≥90%"),
            _mirror("M3", "COMMON", 78.0, m3, ex, ez, (10.0, 10.0), "抬升光束"),
            _mirror("M4", "COMMON", 86.0, m4, ez, near_pass_direction, (10.0, 10.0), "对准倍率切换光轴"),
            _transmissive("FIBE", "MAGNIFICATION_SWITCH", "翻入式柱面扩束转塔", 92.0, fibe, near_pass_direction, (12.0, 12.0), "倍率 A/B 切换", switch_state="A: 1.44 um×5.2 mm；B: 1.05 um×2.667 mm"),
            LayoutElement3D(
                "PRISM_DIVERTER",
                "BRANCH_SWITCH",
                "棱镜切换器",
                100.0,
                _tuple(prism),
                _tuple(near_pass_direction),
                _tuple(reflection_normal(near_pass_direction, prism_to_m5o)),
                12.0,
                12.0,
                "近法向直通/斜入射偏转",
                "N: 直通；O: 偏转至 -Y",
                "分支点坐标为工程基准",
                "表中法向对应 O 偏转状态；N 状态保持入射方向",
            ),
        )
    )

    elements.extend(
        (
            _transmissive("NEAR_E1", "NEAR_NORMAL", "倍率相关扩束等效组", 110.0, near_e1, near_pass_direction, (8.0, 8.0), "近法向支路第一等效焦度", switch_state="随 A/B 档切换", note="焦距见 prescription matrix"),
            _mirror("M5N", "NEAR_NORMAL", 145.0, m5n, near_pass_direction, near_fold_direction, (10.0, 10.0), "近法向后端对准"),
            _transmissive("NORMAL_LFC", "NEAR_NORMAL", "线形成柱面镜+弱 Y 校正组", 166.0, normal_lfc, near_fold_direction, (12.0, 8.0), "形成 X 窄线并校正 Y 长轴", switch_state="随 A/B 档切换", note="最大计算光束 6.41×3.81 mm"),
            _mirror("M6N", "NEAR_NORMAL", 170.0, m6n, near_fold_direction, near_direction, (10.0, 10.0), "折转到物镜照明轴"),
            _transmissive("FOURIER_FILTER", "NEAR_NORMAL", "Fourier 滤波面", 171.0, fourier, near_direction, (8.0, 10.0), "空间频率滤波/杂散光抑制", basis="按参考图保留，参数待确认"),
            _transmissive("AF_BEAMSPLITTER", "NEAR_NORMAL", "自动对焦分光片", 172.0, af_beamsplitter, near_direction, (8.0, 10.0), "照明与 AF/收集通道合束", basis="按参考图保留，参数待确认"),
            _transmissive("OBJECTIVE_IDEAL", "NEAR_NORMAL", "理想薄透镜物镜", 174.0, objective, near_direction, (6.0, 10.0), "近法向照明末级聚焦", basis="f=30 mm 已确认；位置预估", note="固定等效焦距 +30 mm；距晶圆 6 mm"),
        )
    )

    elements.extend(
        (
            _mirror("M5O", "OBLIQUE", 104.0, m5o, prism_to_m5o, oblique_mid_direction, (10.0, 10.0), "OBE 前端对准"),
            _transmissive("OBE_E1", "OBLIQUE", "斜入射非对称扩束等效组", 110.0, obe_e1, oblique_mid_direction, (8.0, 8.0), "斜支路第一等效焦度", switch_state="随 A/B 档切换", note="焦距见 prescription matrix"),
            _mirror("M6O", "OBLIQUE", 145.0, m6o, oblique_mid_direction, oblique_down_direction, (14.0, 10.0), "OBE 后端对准"),
            _mirror("M7O", "OBLIQUE", 160.0, m7o, oblique_down_direction, oblique_direction, (16.0, 10.0), "设置 75°入射角和晶圆交点"),
            _transmissive("OBLIQUE_LFC", "OBLIQUE", "斜入射线形成柱面镜组", 174.0, oblique_lfc, oblique_direction, (16.0, 10.0), "75°支路末级线聚焦", switch_state="随 A/B 档切换", note="最大计算光束 11.98×5.20 mm"),
        )
    )

    # 晶圆和收集通道坐标不重复计入 180 mm 照明光程。
    elements.append(
        LayoutElement3D(
            "WAFER",
            "TARGET",
            "晶圆面",
            180.0,
            (0.0, 0.0, 0.0),
            None,
            (0.0, 0.0, 1.0),
            20.0,
            20.0,
            "两支路共用照明交点",
            "N: 1°；O: 75°",
            "原点和法向定义",
            "光斑尺寸为晶圆面 1/e² 投影尺寸",
        )
    )
    collection_distances = (14.0, 20.0, 26.0, 32.0, 40.0)
    collection_names = (
        "COLLECTION_LENS_1",
        "COLLECTION_LENS_2",
        "COLLECTION_LENS_3",
        "COLLECTION_LENS_4",
        "CAMERA_SENSOR",
    )
    collection_types = ("收集透镜", "收集透镜", "收集透镜", "成像透镜", "相机传感器")
    for distance, name, kind in zip(collection_distances, collection_names, collection_types):
        position = wafer - distance * near_direction
        elements.append(
            LayoutElement3D(
                name,
                "COLLECTION",
                kind,
                None,
                _tuple(position),
                _tuple(-near_direction),
                _tuple(-near_direction),
                12.0 if name != "CAMERA_SENSOR" else 10.0,
                12.0 if name != "CAMERA_SENSOR" else 10.0,
                "晶圆缺陷信号收集/成像",
                "固定",
                "轴线由图确认；间距与处方预估",
                "不计入照明 180 mm 光程",
            )
        )

    common_vertices = (
        ("LASER_OUTPUT", 0.0, laser_output),
        ("M0", 6.0, m0),
        ("ATTENUATOR", 12.0, attenuator),
        ("SET_POLARIZER", 18.0, set_polarizer),
        ("M2", 26.0, m2),
        ("SERVO_BS1", 32.0, bs1),
        ("SERVO_BS2", 36.0, bs2),
        ("ILLUMINATION_POLARIZER", 44.0, illumination_polarizer),
        ("SWATHING_SHUTTER", 52.0, shutter),
        ("WAIST_RELAY_1", 60.0, waist_1),
        ("WAIST_RELAY_2", 72.0, waist_2),
        ("HOMOGENIZER_Y", 75.0, homogenizer),
        ("M3", 78.0, m3),
        ("M4", 86.0, m4),
        ("FIBE", 92.0, fibe),
        ("PRISM_DIVERTER", 100.0, prism),
    )
    near_tail = (
        ("NEAR_E1", 110.0, near_e1),
        ("M5N", 145.0, m5n),
        ("NORMAL_LFC", 166.0, normal_lfc),
        ("M6N", 170.0, m6n),
        ("FOURIER_FILTER", 171.0, fourier),
        ("AF_BEAMSPLITTER", 172.0, af_beamsplitter),
        ("OBJECTIVE_IDEAL", 174.0, objective),
        ("WAFER", 180.0, wafer),
    )
    oblique_tail = (
        ("M5O", 104.0, m5o),
        ("OBE_E1", 110.0, obe_e1),
        ("M6O", 145.0, m6o),
        ("M7O", 160.0, m7o),
        ("OBLIQUE_LFC", 174.0, oblique_lfc),
        ("WAFER", 180.0, wafer),
    )

    near_path = tuple(
        PathVertex3D("NEAR_NORMAL", index, name, s_mm, _tuple(position))
        for index, (name, s_mm, position) in enumerate(common_vertices + near_tail)
    )
    oblique_path = tuple(
        PathVertex3D("OBLIQUE", index, name, s_mm, _tuple(position))
        for index, (name, s_mm, position) in enumerate(common_vertices + oblique_tail)
    )
    layout = Layout3D(
        elements=tuple(elements),
        near_path=near_path,
        oblique_path=oblique_path,
        near_direction=_tuple(near_direction),
        oblique_direction=_tuple(oblique_direction),
    )
    validate_layout_3d(layout)
    return layout


def validate_layout_3d(layout: Layout3D, tolerance_mm: float = 1e-6) -> None:
    """验证两条路径相邻三维距离等于累计光程差。"""

    for path in (layout.near_path, layout.oblique_path):
        if abs(path[0].s_mm) > tolerance_mm or abs(path[-1].s_mm - 180.0) > tolerance_mm:
            raise ValueError("三维路径必须从 s=0 延伸至 s=180 mm。")
        for first, second in zip(path, path[1:]):
            geometric = float(
                np.linalg.norm(np.asarray(second.position_mm) - np.asarray(first.position_mm))
            )
            expected = second.s_mm - first.s_mm
            if abs(geometric - expected) > tolerance_mm:
                raise ValueError(
                    "{}: {}→{} 几何距离 {:.9f} mm 与光程差 {:.9f} mm 不一致。".format(
                        path[0].path_id,
                        first.element_id,
                        second.element_id,
                        geometric,
                        expected,
                    )
                )

