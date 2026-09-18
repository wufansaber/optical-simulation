"""导出完整三维预布局、元件坐标和四工况等效处方。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optical_sim import (
    default_dual_branch_requirements,
    default_layout_3d,
    mirror_incidence_deg,
    solve_equivalent_branch,
    yaw_pitch_deg,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "dual_branch_180mm"
DOC_ASSETS = ROOT / "docs" / "assets" / "illumination-layout"


def _round(value: float | None, digits: int = 6):
    return "" if value is None else round(value, digits)


def _local_transverse_axes(direction):
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    reference = np.asarray((0.0, 1.0, 0.0))
    if abs(float(np.dot(reference, direction))) > 0.95:
        reference = np.asarray((0.0, 0.0, 1.0))
    local_y = reference - float(np.dot(reference, direction)) * direction
    local_y = local_y / np.linalg.norm(local_y)
    local_x = np.cross(local_y, direction)
    local_x = local_x / np.linalg.norm(local_x)
    return local_x, local_y


def _coordinate_rows(layout):
    rows = []
    for element in layout.elements:
        axis_yaw = axis_pitch = normal_yaw = normal_pitch = incidence = None
        local_x = local_y = None
        if element.chief_ray_direction is not None:
            axis_yaw, axis_pitch = yaw_pitch_deg(element.chief_ray_direction)
            local_x, local_y = _local_transverse_axes(element.chief_ray_direction)
        if element.surface_normal is not None:
            normal_yaw, normal_pitch = yaw_pitch_deg(element.surface_normal)
        if element.component_type == "平面反射镜":
            incidence = float(element.note.split(" ")[1])
        if element.element_id in {"FIBE", "NEAR_E1", "NORMAL_LFC", "OBE_E1", "OBLIQUE_LFC"}:
            cylinder_axis = "主柱面轴=local_y；正交弱校正轴=local_x"
        elif element.element_id == "HOMOGENIZER_Y":
            cylinder_axis = "匀化长轴=local_y"
        else:
            cylinder_axis = "轴对称/滚转不敏感" if element.chief_ray_direction is not None else "不适用"
        rows.append(
            {
                "element_id": element.element_id,
                "path_group": element.path_group,
                "component_type": element.component_type,
                "s_mm": _round(element.s_mm),
                "x_mm": _round(element.position_mm[0]),
                "y_mm": _round(element.position_mm[1]),
                "z_mm": _round(element.position_mm[2]),
                "chief_dx": _round(None if element.chief_ray_direction is None else element.chief_ray_direction[0]),
                "chief_dy": _round(None if element.chief_ray_direction is None else element.chief_ray_direction[1]),
                "chief_dz": _round(None if element.chief_ray_direction is None else element.chief_ray_direction[2]),
                "chief_yaw_deg": _round(axis_yaw),
                "chief_pitch_deg": _round(axis_pitch),
                "local_x_dx": _round(None if local_x is None else local_x[0]),
                "local_x_dy": _round(None if local_x is None else local_x[1]),
                "local_x_dz": _round(None if local_x is None else local_x[2]),
                "local_y_dx": _round(None if local_y is None else local_y[0]),
                "local_y_dy": _round(None if local_y is None else local_y[1]),
                "local_y_dz": _round(None if local_y is None else local_y[2]),
                "cylinder_axis_definition": cylinder_axis,
                "normal_nx": _round(None if element.surface_normal is None else element.surface_normal[0]),
                "normal_ny": _round(None if element.surface_normal is None else element.surface_normal[1]),
                "normal_nz": _round(None if element.surface_normal is None else element.surface_normal[2]),
                "normal_yaw_deg": _round(normal_yaw),
                "normal_pitch_deg": _round(normal_pitch),
                "mirror_incidence_deg": _round(incidence),
                "clear_aperture_x_mm": element.aperture_x_mm,
                "clear_aperture_y_mm": element.aperture_y_mm,
                "switch_state": element.switch_state,
                "function": element.function,
                "basis": element.basis,
                "note": element.note,
            }
        )
    return rows


def _path_rows(layout):
    rows = []
    for path in (layout.near_path, layout.oblique_path):
        for first, second in zip(path, path[1:]):
            start = np.asarray(first.position_mm)
            stop = np.asarray(second.position_mm)
            direction = (stop - start) / np.linalg.norm(stop - start)
            rows.append(
                {
                    "path_id": first.path_id,
                    "segment": "{} -> {}".format(first.element_id, second.element_id),
                    "s_start_mm": first.s_mm,
                    "s_end_mm": second.s_mm,
                    "length_mm": second.s_mm - first.s_mm,
                    "x_start_mm": _round(start[0]),
                    "y_start_mm": _round(start[1]),
                    "z_start_mm": _round(start[2]),
                    "x_end_mm": _round(stop[0]),
                    "y_end_mm": _round(stop[1]),
                    "z_end_mm": _round(stop[2]),
                    "direction_dx": _round(direction[0]),
                    "direction_dy": _round(direction[1]),
                    "direction_dz": _round(direction[2]),
                }
            )
    return rows


def _prescription_rows():
    rows = []
    labels = {
        "near_normal_1deg_ratio_a": "N-A",
        "near_normal_1deg_ratio_b": "N-B",
        "oblique_75deg_ratio_a": "O-A",
        "oblique_75deg_ratio_b": "O-B",
    }
    for requirement in default_dual_branch_requirements():
        result = solve_equivalent_branch(requirement)
        near = requirement.objective_focal_m is not None
        rows.append(
            {
                "case": labels[requirement.name],
                "incidence_deg": requirement.incidence_angle_to_normal_deg,
                "fibe_state": "A" if requirement.name.endswith("ratio_a") else "B",
                "branch_state": "近法向直通" if near else "斜入射偏转",
                "element": "NEAR_E1" if near else "OBE_E1",
                "s_mm": requirement.lens1_s_m * 1e3,
                "fx_mm": result.lenses.focal1_x_m * 1e3,
                "fy_mm": result.lenses.focal1_y_m * 1e3,
                "target_x_um": requirement.target_surface_diameter_x_m * 1e6,
                "target_y_mm": requirement.target_surface_diameter_y_m * 1e3,
                "status": "等效一阶求解值",
            }
        )
        rows.append(
            {
                "case": labels[requirement.name],
                "incidence_deg": requirement.incidence_angle_to_normal_deg,
                "fibe_state": "A" if requirement.name.endswith("ratio_a") else "B",
                "branch_state": "近法向直通" if near else "斜入射偏转",
                "element": "NORMAL_LFC" if near else "OBLIQUE_LFC",
                "s_mm": requirement.lens2_s_m * 1e3,
                "fx_mm": result.lenses.focal2_x_m * 1e3,
                "fy_mm": result.lenses.focal2_y_m * 1e3,
                "target_x_um": requirement.target_surface_diameter_x_m * 1e6,
                "target_y_mm": requirement.target_surface_diameter_y_m * 1e3,
                "status": "等效一阶求解值",
            }
        )
        if near:
            rows.append(
                {
                    "case": labels[requirement.name],
                    "incidence_deg": requirement.incidence_angle_to_normal_deg,
                    "fibe_state": "A" if requirement.name.endswith("ratio_a") else "B",
                    "branch_state": "近法向直通",
                    "element": "OBJECTIVE_IDEAL",
                    "s_mm": requirement.objective_s_m * 1e3,
                    "fx_mm": requirement.objective_focal_m * 1e3,
                    "fy_mm": requirement.objective_focal_m * 1e3,
                    "target_x_um": requirement.target_surface_diameter_x_m * 1e6,
                    "target_y_mm": requirement.target_surface_diameter_y_m * 1e3,
                    "status": "用户确认固定值",
                }
            )
    return rows


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_layout(layout):
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
        }
    )
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_top = fig.add_subplot(grid[0, 1])
    ax_side = fig.add_subplot(grid[1, 1])

    common = [vertex for vertex in layout.near_path if vertex.s_mm <= 100.0]
    near = [vertex for vertex in layout.near_path if vertex.s_mm >= 100.0]
    oblique = [vertex for vertex in layout.oblique_path if vertex.s_mm >= 100.0]
    collection_elements = sorted(
        [element for element in layout.elements if element.path_group == "COLLECTION"],
        key=lambda element: element.position_mm[2],
    )
    wafer = next(element for element in layout.elements if element.element_id == "WAFER")
    collection = [wafer.position_mm] + [element.position_mm for element in collection_elements]

    series = (
        (common, "共用前端", "#3568a8", 2.8),
        (near, "近法向 1°", "#3a9142", 3.0),
        (oblique, "斜入射 75°", "#d84b4b", 3.0),
    )
    for vertices, label, color, width in series:
        xyz = np.asarray([vertex.position_mm for vertex in vertices])
        ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=width, label=label)
        ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=18)
        ax_top.plot(xyz[:, 0], xyz[:, 1], color=color, lw=width, label=label)
        ax_top.scatter(xyz[:, 0], xyz[:, 1], color=color, s=12)
        ax_side.plot(xyz[:, 0], xyz[:, 2], color=color, lw=width, label=label)
        ax_side.scatter(xyz[:, 0], xyz[:, 2], color=color, s=12)

    collection_xyz = np.asarray(collection)
    for axis, a, b in ((ax3d, 0, 2),):
        del axis, a, b
    ax3d.plot(
        collection_xyz[:, 0], collection_xyz[:, 1], collection_xyz[:, 2],
        color="#7a4da1", lw=2.0, ls="--", label="收集/相机轴",
    )
    ax_top.plot(collection_xyz[:, 0], collection_xyz[:, 1], color="#7a4da1", lw=2.0, ls="--")
    ax_side.plot(collection_xyz[:, 0], collection_xyz[:, 2], color="#7a4da1", lw=2.0, ls="--")

    label_text = {
        "LASER_OUTPUT": "Laser out", "M0": "M0", "M2": "M2", "M3": "M3",
        "M4": "M4", "FIBE": "FIBE", "PRISM_DIVERTER": "Prism",
        "M5N": "M5N", "NORMAL_LFC": "N-LFC", "M6N": "M6N",
        "OBJECTIVE_IDEAL": "Obj 30", "M5O": "M5O", "M6O": "M6O",
        "M7O": "M7O", "OBLIQUE_LFC": "O-LFC", "WAFER": "Wafer",
        "CAMERA_SENSOR": "Camera",
    }
    top_offsets = {
        "LASER_OUTPUT": (-12, 5), "M0": (2, -12), "M2": (2, 5), "M3": (-8, -13),
        "M4": (3, 5), "FIBE": (3, 5), "PRISM_DIVERTER": (3, 5),
        "M5N": (3, -14), "NORMAL_LFC": (-28, 5), "M6N": (-12, -14),
        "OBJECTIVE_IDEAL": (-8, 6), "M5O": (3, -14), "M6O": (3, -14),
        "M7O": (3, 5), "OBLIQUE_LFC": (3, 5), "WAFER": (3, -14),
        "CAMERA_SENSOR": (3, 5),
    }
    side_offsets = {
        "LASER_OUTPUT": (-12, 5), "M0": (2, -12), "M2": (2, 5), "M3": (2, -13),
        "M4": (3, 5), "FIBE": (3, 5), "PRISM_DIVERTER": (3, -14),
        "M5N": (3, 5), "NORMAL_LFC": (-32, 6), "M6N": (-20, -14),
        "OBJECTIVE_IDEAL": (3, -13), "M5O": (3, 5), "M6O": (3, 5),
        "M7O": (3, -13), "OBLIQUE_LFC": (-28, 6), "WAFER": (3, -14),
        "CAMERA_SENSOR": (3, 5),
    }
    for element in layout.elements:
        if element.element_id not in label_text:
            continue
        x, y, z = element.position_mm
        label = label_text[element.element_id]
        ax3d.text(x, y, z + 1.2, label, fontsize=7)
        ax_top.annotate(label, (x, y), xytext=top_offsets[element.element_id], textcoords="offset points", fontsize=7)
        ax_side.annotate(label, (x, z), xytext=side_offsets[element.element_id], textcoords="offset points", fontsize=7)

    wafer_x = np.linspace(-7.0, 7.0, 2)
    wafer_y = np.linspace(-7.0, 7.0, 2)
    wx, wy = np.meshgrid(wafer_x, wafer_y)
    wz = np.zeros_like(wx)
    ax3d.plot_surface(wx, wy, wz, alpha=0.18, color="#555555")

    ax3d.set_title("180 mm 双支路照明三维工程预布局")
    ax3d.set_xlabel("X / mm")
    ax3d.set_ylabel("Y / mm")
    ax3d.set_zlabel("Z / mm")
    ax3d.view_init(elev=25, azim=-58)
    ax3d.set_box_aspect((1.7, 1.0, 0.65))
    ax3d.legend(loc="upper left")

    ax_top.set_title("俯视图（X-Y）")
    ax_top.set_xlabel("X / mm")
    ax_top.set_ylabel("Y / mm")
    ax_top.axis("equal")
    ax_top.grid(True, alpha=0.25)
    ax_side.set_title("侧视图（X-Z）")
    ax_side.set_xlabel("X / mm")
    ax_side.set_ylabel("Z / mm")
    ax_side.axis("equal")
    ax_side.grid(True, alpha=0.25)
    fig.suptitle(
        "晶圆中心=(0,0,0)，+Z=晶圆外法线；坐标为工程预估，光程和入射角严格闭合",
        fontsize=11,
    )
    return fig


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOC_ASSETS.mkdir(parents=True, exist_ok=True)
    layout = default_layout_3d()
    coordinate_rows = _coordinate_rows(layout)
    path_rows = _path_rows(layout)
    prescription_rows = _prescription_rows()
    _write_csv(OUTPUT / "component_coordinates_3d.csv", coordinate_rows)
    _write_csv(OUTPUT / "raypath_segments_3d.csv", path_rows)
    _write_csv(OUTPUT / "component_prescription_matrix.csv", prescription_rows)

    payload = {
        "coordinate_system": {
            "origin": "wafer illumination intersection",
            "units": "mm",
            "x_axis": "75-degree beam projection direction on wafer",
            "y_axis": "wafer long-line direction; right-handed with X/Z",
            "z_axis": "wafer outward normal",
        },
        "confirmed_constraints": {
            "wavelength_nm": 355.0,
            "laser_model": "Paladin Advanced 355-8000",
            "laser_mechanical_length_mm": 90.0,
            "illumination_path_from_output_to_wafer_mm": 180.0,
            "near_incidence_deg": 1.0,
            "oblique_incidence_deg": 75.0,
            "objective_efl_mm": 30.0,
            "spot_definition": "wafer-plane 1/e^2 full width",
        },
        "elements": [asdict(element) for element in layout.elements],
        "near_path": [asdict(vertex) for vertex in layout.near_path],
        "oblique_path": [asdict(vertex) for vertex in layout.oblique_path],
        "prescription_matrix": prescription_rows,
    }
    with (OUTPUT / "layout_3d.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    figure = _plot_layout(layout)
    for suffix in ("png", "svg"):
        figure.savefig(OUTPUT / ("complete_3d_layout." + suffix), dpi=220 if suffix == "png" else None)
    figure.savefig(DOC_ASSETS / "complete_3d_layout.png", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
