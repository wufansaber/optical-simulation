"""绘制四工况等效薄透镜计算光线追迹。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from optical_sim import (
    Paladin355Input,
    default_dual_branch_requirements,
    sample_branch_envelope,
    sample_paraxial_ray_bundle,
    solve_equivalent_branch,
)


OUTPUT_DIRECTORY = Path("outputs/dual_branch_180mm")


def _configure_fonts() -> None:
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def _case_label(requirement) -> str:
    branch = "O" if requirement.incidence_angle_to_normal_deg > 45.0 else "N"
    ratio = "A" if requirement.name.endswith("ratio_a") else "B"
    return "{}-{}".format(branch, ratio)


def _target_text(requirement) -> str:
    return "{:.2f} μm × {:.3f} mm".format(
        requirement.target_surface_diameter_x_m * 1e6,
        requirement.target_surface_diameter_y_m * 1e3,
    )


def _plot_axis_trace(axis_plot, result, direction, row_index, branch_color):
    s, rays = sample_paraxial_ray_bundle(
        result,
        direction,
        ray_count=17,
        samples_per_segment=1200,
    )
    envelope_s, radius_x, radius_y = sample_branch_envelope(result, samples_per_segment=1200)
    radius = radius_x if direction == "x" else radius_y
    s_mm = s * 1e3
    rays_mm = rays * 1e3
    radius_mm = radius * 1e3

    axis_plot.axvspan(0.0, 100.0, color="#DCEAF7", alpha=0.40, linewidth=0)
    axis_plot.axvspan(100.0, 180.0, color=branch_color, alpha=0.08, linewidth=0)
    axis_plot.fill_between(
        envelope_s * 1e3,
        -radius_mm,
        radius_mm,
        color="#6F42C1",
        alpha=0.10,
        label="1/e²包络" if row_index == 0 else None,
    )
    for ray_index in range(rays.shape[1] - 1):
        axis_plot.plot(s_mm, rays_mm[:, ray_index], color="#6F42C1", alpha=0.48, linewidth=0.55)
    axis_plot.plot(s_mm, rays_mm[:, -1], color="#202124", linewidth=1.0, label="中心主光线" if row_index == 0 else None)

    optical_planes = [
        (92.0, "FIBE", ":"),
        (100.0, "Prism", "--"),
        (110.0, "E1", "-"),
    ]
    if result.requirement.objective_s_m is not None:
        optical_planes.extend(
            [
                (result.requirement.lens2_s_m * 1e3, "LFC", "-"),
                (result.requirement.objective_s_m * 1e3, "Obj30", "-."),
            ]
        )
    else:
        optical_planes.append((result.requirement.lens2_s_m * 1e3, "E2", "-"))
    optical_planes.append((180.0, "Wafer", "-"))
    for position, label, style in optical_planes:
        color = "#E59D28" if position in (92.0, 100.0) else "#555555"
        axis_plot.axvline(position, color=color, linestyle=style, linewidth=0.9, alpha=0.9)
        if row_index == 0:
            axis_plot.text(
                position,
                1.01,
                label,
                transform=axis_plot.get_xaxis_transform(),
                ha="center" if position < 180.0 else "right",
                va="bottom",
                fontsize=7.2,
                color=color,
            )

    limit = max(float(np.max(radius_mm)) * 1.12, 0.001)
    axis_plot.set_xlim(0.0, 181.0)
    axis_plot.set_ylim(-limit, limit)
    axis_plot.grid(True, alpha=0.20, linewidth=0.6)
    axis_plot.set_ylabel("{}向光线高度 (mm)".format(direction.upper()))
    axis_plot.set_xlabel("展开中心光程 s (mm)")

    if direction == "x":
        lens = result.lenses
        axis_plot.text(
            0.015,
            0.94,
            "E1 f={:.3f} mm   {} f={:.3f} mm{}".format(
                lens.focal1_x_m * 1e3,
                "LFC" if result.requirement.objective_s_m is not None else "E2",
                lens.focal2_x_m * 1e3,
                "   Objective f=30.000 mm" if result.requirement.objective_s_m is not None else "",
            ),
            transform=axis_plot.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#40464D",
        )
        inset = inset_axes(axis_plot, width="34%", height="42%", loc="lower left", borderpad=1.0)
        mask = s >= (result.requirement.total_path_m - 50e-6)
        distance_um = (s[mask] - result.requirement.total_path_m) * 1e6
        for ray_index in range(rays.shape[1] - 1):
            inset.plot(distance_um, rays[mask, ray_index] * 1e6, color="#6F42C1", alpha=0.55, linewidth=0.55)
        inset.plot(distance_um, rays[mask, -1] * 1e6, color="#202124", linewidth=0.9)
        inset.axvline(0.0, color="#202124", linewidth=0.8)
        inset.set_title("晶圆前50 μm放大", fontsize=7.0)
        inset.set_xlabel("距晶圆 (μm)", fontsize=6.5)
        inset.set_ylabel("X (μm)", fontsize=6.5)
        inset.tick_params(labelsize=6.2)
        inset.grid(True, alpha=0.18, linewidth=0.4)
    else:
        lens = result.lenses
        axis_plot.text(
            0.015,
            0.94,
            "E1 f={:.3f} mm   {} f={:.3f} mm{}".format(
                lens.focal1_y_m * 1e3,
                "LFC/Y补偿" if result.requirement.objective_s_m is not None else "E2",
                lens.focal2_y_m * 1e3,
                "   Objective f=30.000 mm" if result.requirement.objective_s_m is not None else "",
            ),
            transform=axis_plot.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#40464D",
        )
    return s, rays


def render() -> None:
    _configure_fonts()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    laser = Paladin355Input()
    requirements = default_dual_branch_requirements()
    results = [solve_equivalent_branch(requirement, laser) for requirement in requirements]
    figure, axes = plt.subplots(4, 2, figsize=(18, 18), constrained_layout=False)
    trace_payload = {}

    for row_index, result in enumerate(results):
        requirement = result.requirement
        branch_color = "#59A14F" if requirement.incidence_angle_to_normal_deg < 45.0 else "#E15759"
        for column_index, direction in enumerate(("x", "y")):
            axis_plot = axes[row_index, column_index]
            s, rays = _plot_axis_trace(axis_plot, result, direction, row_index, branch_color)
            trace_payload["{}_{}_s_m".format(_case_label(requirement), direction)] = s
            trace_payload["{}_{}_height_m".format(_case_label(requirement), direction)] = rays
            if column_index == 0:
                axis_plot.set_title(
                    "{}｜{}°｜目标 {}｜X入射面追迹".format(
                        _case_label(requirement),
                        int(requirement.incidence_angle_to_normal_deg),
                        _target_text(requirement),
                    ),
                    fontsize=10.5,
                )
            else:
                axis_plot.set_title(
                    "{}｜Y长轴追迹".format(_case_label(requirement)),
                    fontsize=10.5,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.045, top=0.90, hspace=0.31, wspace=0.09)
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.938))
    figure.suptitle(
        "四工况计算光线追迹｜355 nm｜M²=1.2｜展开几何光程180 mm",
        fontsize=16,
        y=0.982,
    )
    figure.text(
        0.5,
        0.958,
        "近法向：E1→Normal LFC/补偿面→30 mm理想物镜→晶圆；斜入射：E1→E2→晶圆。紫色阴影为q参数1/e²包络。",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#4F5963",
    )

    for suffix in ("png", "svg", "pdf"):
        figure.savefig(
            OUTPUT_DIRECTORY / ("computed_ray_trace." + suffix),
            dpi=240 if suffix == "png" else None,
            facecolor="white",
        )
    np.savez_compressed(OUTPUT_DIRECTORY / "computed_ray_trace_data.npz", **trace_payload)
    plt.close(figure)


if __name__ == "__main__":
    render()
