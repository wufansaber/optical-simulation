"""运行 180 mm、两角度、两倍率的四工况等效光路初算。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optical_sim import (
    IlluminationPerformanceRequirement,
    Paladin355Input,
    default_dual_branch_requirements,
    default_efficiency_budget,
    default_estimated_layout,
    evaluate_case_performance,
    sample_branch_envelope,
    solve_equivalent_branch,
)


OUTPUT_DIRECTORY = Path("outputs/dual_branch_180mm")


COMMON_FIXED_PATH = (
    "Laser -> M0 -> Attenuator -> Set Polarizer -> M2 -> Servo/D1/D2 -> "
    "Illumination Polarizer -> Swathing Shutter -> Waist Relay -> Y Homogenizer -> M3 -> M4"
)


def _case_label(requirement):
    branch = "O" if requirement.incidence_angle_to_normal_deg > 45.0 else "N"
    ratio = "A" if requirement.name.endswith("ratio_a") else "B"
    return "{}-{}".format(branch, ratio)


def _case_path_row(requirement):
    is_oblique = requirement.incidence_angle_to_normal_deg > 45.0
    ratio = "A" if requirement.name.endswith("ratio_a") else "B"
    return {
        "case": requirement.name,
        "incidence_angle_deg": requirement.incidence_angle_to_normal_deg,
        "common_fixed_or_adjustable": COMMON_FIXED_PATH,
        "shared_magnification_switch": "FIBE / beam-expander ratio {}".format(ratio),
        "branch_selector": (
            "Prism Diverter = deflect to oblique branch"
            if is_oblique
            else "Prism Diverter = near-normal pass state"
        ),
        "exclusive_branch": (
            "M5O -> OBE -> M6O -> M7O -> Oblique LFC -> wafer"
            if is_oblique
            else "Near-normal expander -> M5N -> Normal LFC -> M6N -> Objective -> wafer"
        ),
    }


def _summary_row(result):
    requirement = result.requirement
    lenses = result.lenses
    return {
        "case": requirement.name,
        "incidence_angle_to_normal_deg": requirement.incidence_angle_to_normal_deg,
        "total_path_mm": requirement.total_path_m * 1e3,
        "target_surface_x_um": requirement.target_surface_diameter_x_m * 1e6,
        "target_surface_y_mm": requirement.target_surface_diameter_y_m * 1e3,
        "output_surface_x_um": result.output_surface_diameter_x_m * 1e6,
        "output_surface_y_mm": result.output_surface_diameter_y_m * 1e3,
        "output_normal_x_um": result.output_normal_diameter_x_m * 1e6,
        "output_normal_y_mm": result.output_normal_diameter_y_m * 1e3,
        "lens1_s_mm": lenses.lens1_s_m * 1e3,
        "lens2_s_mm": lenses.lens2_s_m * 1e3,
        "objective_s_mm": (
            requirement.objective_s_m * 1e3
            if requirement.objective_s_m is not None
            else ""
        ),
        "objective_focal_mm": (
            requirement.objective_focal_m * 1e3
            if requirement.objective_focal_m is not None
            else ""
        ),
        "lens1_fx_mm": lenses.focal1_x_m * 1e3,
        "lens1_fy_mm": lenses.focal1_y_m * 1e3,
        "lens2_fx_mm": lenses.focal2_x_m * 1e3,
        "lens2_fy_mm": lenses.focal2_y_m * 1e3,
        "paraxial_slope_x": result.paraxial_slope_x,
        "paraxial_slope_y": result.paraxial_slope_y,
        "geometric_na_x": result.geometric_na_x,
        "geometric_na_y": result.geometric_na_y,
        "paraxial_valid_x": result.paraxial_valid_x,
        "paraxial_valid_y": result.paraxial_valid_y,
        "ideal_peak_irradiance_w_m2": result.ideal_peak_irradiance_w_m2,
    }


def _performance_row(requirement, result):
    return {
        "case": _case_label(requirement),
        "case_name": requirement.name,
        "uniformity_model_percent": 100.0 * result.uniformity_model_ratio,
        "uniformity_allocated_minimum_percent": 100.0 * result.uniformity_allocated_minimum_ratio,
        "path_efficiency_nominal_percent": 100.0 * result.path_efficiency_nominal_ratio,
        "path_efficiency_allocated_minimum_percent": 100.0 * result.path_efficiency_allocated_minimum_ratio,
        "target_capture_percent": 100.0 * result.target_capture_ratio,
        "useful_efficiency_nominal_percent": 100.0 * result.useful_efficiency_nominal_ratio,
        "useful_efficiency_allocated_minimum_percent": 100.0 * result.useful_efficiency_allocated_minimum_ratio,
        "uniformity_pass": result.uniformity_pass,
        "efficiency_pass": result.efficiency_pass,
    }


def _plot(results, output_path: Path) -> None:
    performance = IlluminationPerformanceRequirement()
    figure, axes = plt.subplots(len(results), 2, figsize=(14, 4.2 * len(results)))
    colors = {"x": "#C44E52", "y": "#4C72B0"}
    for row_index, result in enumerate(results):
        requirement = result.requirement
        s, radius_x, radius_y = sample_branch_envelope(result)
        envelope_axis = axes[row_index, 0]
        envelope_axis.plot(s * 1e3, 2.0 * radius_x * 1e3, color=colors["x"], label="X beam-normal diameter")
        envelope_axis.plot(s * 1e3, 2.0 * radius_y * 1e3, color=colors["y"], label="Y beam-normal diameter")
        envelope_axis.axvline(result.lenses.lens1_s_m * 1e3, color="#666666", linestyle="--", linewidth=1)
        envelope_axis.axvline(result.lenses.lens2_s_m * 1e3, color="#666666", linestyle="--", linewidth=1)
        if requirement.objective_s_m is not None:
            envelope_axis.axvline(
                requirement.objective_s_m * 1e3,
                color="#2E8B57",
                linestyle="-.",
                linewidth=1.1,
            )
        envelope_axis.axvline(requirement.total_path_m * 1e3, color="#222222", linestyle=":", linewidth=1.2)
        envelope_axis.set_yscale("log")
        envelope_axis.set_xlabel("Unfolded centerline path s (mm)")
        envelope_axis.set_ylabel("1/e^2 full diameter (mm)")
        envelope_axis.set_title(
            "{} | angle {:.1f} deg | path {:.1f} mm".format(
                requirement.name,
                requirement.incidence_angle_to_normal_deg,
                requirement.total_path_m * 1e3,
            )
        )
        envelope_axis.grid(True, which="both", alpha=0.25)
        envelope_axis.legend(loc="best", fontsize=8)

        surface_radius_x = 0.5 * result.output_surface_diameter_x_m
        surface_radius_y = 0.5 * result.output_surface_diameter_y_m
        x_um = np.linspace(-2.0 * surface_radius_x, 2.0 * surface_radius_x, 401) * 1e6
        y_mm = np.linspace(-2.0 * surface_radius_y, 2.0 * surface_radius_y, 401) * 1e3
        x_grid, y_grid = np.meshgrid(x_um * 1e-6, y_mm * 1e-3)
        normalized = np.exp(
            -2.0 * (x_grid / surface_radius_x) ** 2
            -2.0 * np.abs(y_grid / surface_radius_y) ** performance.design_supergaussian_order_y
        )
        field_axis = axes[row_index, 1]
        image = field_axis.imshow(
            normalized,
            origin="lower",
            extent=[x_um[0], x_um[-1], y_mm[0], y_mm[-1]],
            aspect="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        field_axis.contour(x_um, y_mm, normalized, levels=[np.exp(-2.0)], colors="cyan", linewidths=1.0)
        field_axis.set_xlabel("Wafer X (um)")
        field_axis.set_ylabel("Wafer Y (mm)")
        field_axis.set_title(
            "Target {:.3f} um x {:.3f} mm | Gaussian X / SG36 Y".format(
                requirement.target_surface_diameter_x_m * 1e6,
                requirement.target_surface_diameter_y_m * 1e3,
            )
        )
        figure.colorbar(image, ax=field_axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        "Equivalent dual-branch Gaussian model\n"
        "Near-normal: E1=110 mm, LFC=166 mm, ideal objective f=30 mm at 174 mm; oblique: E1=110 mm, E2=174 mm",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output_path, dpi=240)
    plt.close(figure)


def _plot_performance_compliance(requirements, performance_results, output_path: Path) -> None:
    labels = [_case_label(item) for item in requirements]
    x = np.arange(len(labels))
    width = 0.34
    requirement = IlluminationPerformanceRequirement()
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    uniformity_model = [100.0 * item.uniformity_model_ratio for item in performance_results]
    uniformity_minimum = [100.0 * item.uniformity_allocated_minimum_ratio for item in performance_results]
    axes[0].bar(x - width / 2, uniformity_model, width, label="Model nominal", color="#4C78A8")
    axes[0].bar(x + width / 2, uniformity_minimum, width, label="Allocated minimum", color="#72B7B2")
    axes[0].axhline(100.0 * requirement.minimum_uniformity_ratio, color="#E15759", linestyle="--", label="Requirement 90%")
    axes[0].set_ylim(85.0, 100.0)
    axes[0].set_ylabel("Y-line uniformity Imin/Imax (%)")
    axes[0].set_title("Uniformity over central 90% of Y length")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)

    efficiency_nominal = [100.0 * item.useful_efficiency_nominal_ratio for item in performance_results]
    efficiency_minimum = [100.0 * item.useful_efficiency_allocated_minimum_ratio for item in performance_results]
    axes[1].bar(x - width / 2, efficiency_nominal, width, label="Budget nominal", color="#F2A541")
    axes[1].bar(x + width / 2, efficiency_minimum, width, label="Allocated minimum", color="#B279A2")
    axes[1].axhline(100.0 * requirement.minimum_useful_efficiency_ratio, color="#E15759", linestyle="--", label="Requirement 65%")
    axes[1].set_ylim(60.0, 85.0)
    axes[1].set_ylabel("Useful optical efficiency (%)")
    axes[1].set_title("Laser output to target rectangle on wafer")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    for axis, series in ((axes[0], uniformity_minimum), (axes[1], efficiency_minimum)):
        for index, value in enumerate(series):
            axis.text(index + width / 2, value + 0.35, "{:.1f}%".format(value), ha="center", va="bottom", fontsize=8)

    figure.suptitle("Illumination performance compliance for four operating cases", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=240)
    plt.close(figure)


def _plot_path_topology(requirements, output_path: Path) -> None:
    """Show the shared path and the two independent switching dimensions."""

    figure, axis = plt.subplots(figsize=(15, 8.2))
    colors = {
        "common": "#4C78A8",
        "ratio_a": "#F2A541",
        "ratio_b": "#9C6ADE",
        "near_normal": "#59A14F",
        "oblique": "#E15759",
    }
    y_positions = np.arange(len(requirements))[::-1]
    for y, requirement in zip(y_positions, requirements):
        ratio = "ratio_a" if requirement.name.endswith("ratio_a") else "ratio_b"
        branch = "oblique" if requirement.incidence_angle_to_normal_deg > 45.0 else "near_normal"
        axis.plot([0, 92], [y, y], color=colors["common"], linewidth=12, solid_capstyle="butt")
        axis.plot([92, 100], [y, y], color=colors[ratio], linewidth=12, solid_capstyle="butt")
        axis.plot([100, 180], [y, y], color=colors[branch], linewidth=12, solid_capstyle="butt")
        axis.scatter([100], [y], s=130, marker="D", color="#222222", zorder=4)
        axis.scatter([180], [y], s=130, marker="|", color="#222222", zorder=4)
        axis.text(46, y + 0.18, "COMMON", ha="center", va="bottom", fontsize=9, color=colors["common"])
        axis.text(96, y + 0.18, "FIBE {}".format("A" if ratio == "ratio_a" else "B"), ha="center", va="bottom", fontsize=9)
        axis.text(140, y + 0.18, "{} ONLY".format("OBLIQUE" if branch == "oblique" else "NEAR-NORMAL"), ha="center", va="bottom", fontsize=9, color=colors[branch])

    axis.axvline(92, color="#777777", linestyle=":", linewidth=1)
    axis.axvline(100, color="#222222", linestyle="--", linewidth=1)
    axis.text(90.5, len(requirements) - 0.35, "magnification switch", ha="right", va="bottom", fontsize=9)
    axis.text(101.5, len(requirements) - 0.35, "Prism branch switch", ha="left", va="bottom", fontsize=9)
    axis.text(180, len(requirements) - 0.35, "wafer", ha="center", va="bottom", fontsize=9)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [
            "{} | {:.0f} deg | {}".format(
                ("O" if requirement.incidence_angle_to_normal_deg > 45.0 else "N")
                + ("-A" if requirement.name.endswith("ratio_a") else "-B"),
                requirement.incidence_angle_to_normal_deg,
                "1.44 um x 5.2 mm" if requirement.name.endswith("ratio_a") else "1.05 um x 2.667 mm",
            )
            for requirement in requirements
        ]
    )
    axis.set_xlim(-2, 184)
    axis.set_ylim(-1.2, len(requirements) - 0.05)
    axis.set_xlabel("Estimated unfolded geometric path s (mm)")
    axis.set_title("Four operating cases: shared path, magnification switch, and branch switch", pad=24)
    axis.text(
        0,
        -0.60,
        "COMMON: Laser - M0 - Attenuator - Set Polarizer - M2 - Servo/D1/D2 - Polarizer - Shutter - Relay - Y Homogenizer - M3 - M4",
        ha="left",
        va="center",
        fontsize=8.5,
        color=colors["common"],
    )
    axis.text(
        0,
        -0.85,
        "NEAR: fixed expander - M5N - Normal LFC - M6N - Objective    |    OBLIQUE: M5O - OBE - M6O - M7O - Oblique LFC",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    axis.grid(axis="x", alpha=0.2)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=240)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    laser = Paladin355Input()
    requirements = default_dual_branch_requirements()
    results = [solve_equivalent_branch(requirement, laser) for requirement in requirements]
    performance_requirement = IlluminationPerformanceRequirement()
    performance_results = [
        evaluate_case_performance(requirement, performance_requirement)
        for requirement in requirements
    ]

    performance_rows = [
        _performance_row(requirement, performance_result)
        for requirement, performance_result in zip(requirements, performance_results)
    ]
    performance_path = OUTPUT_DIRECTORY / "performance_summary.csv"
    with performance_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(performance_rows[0]))
        writer.writeheader()
        writer.writerows(performance_rows)

    efficiency_budget_path = OUTPUT_DIRECTORY / "efficiency_budget.csv"
    with efficiency_budget_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = ["case", "group", "element", "nominal_percent", "allocated_minimum_percent", "note"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for requirement in requirements:
            for item in default_efficiency_budget(requirement):
                writer.writerow(
                    {
                        "case": _case_label(requirement),
                        "group": item.group,
                        "element": item.element,
                        "nominal_percent": 100.0 * item.nominal_ratio,
                        "allocated_minimum_percent": 100.0 * item.allocated_minimum_ratio,
                        "note": item.note,
                    }
                )

    path_matrix_path = OUTPUT_DIRECTORY / "case_path_matrix.csv"
    path_rows = [_case_path_row(requirement) for requirement in requirements]
    with path_matrix_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)

    layout_path = OUTPUT_DIRECTORY / "estimated_layout.csv"
    with layout_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["branch", "element", "s_mm", "role"])
        writer.writeheader()
        for element in default_estimated_layout():
            writer.writerow(
                {
                    "branch": element.branch,
                    "element": element.element,
                    "s_mm": element.s_m * 1e3,
                    "role": element.role,
                }
            )

    summary_rows = [_summary_row(result) for result in results]
    summary_path = OUTPUT_DIRECTORY / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = OUTPUT_DIRECTORY / "results.json"
    oblique_results = [
        result
        for result in results
        if result.requirement.incidence_angle_to_normal_deg == 75.0
    ]
    beam_diameters_at_lfc = []
    for result in oblique_results:
        s_values, radius_x, radius_y = sample_branch_envelope(result)
        lens_index = int(np.argmin(np.abs(s_values - result.lenses.lens2_s_m)))
        beam_diameters_at_lfc.append(
            (2.0 * radius_x[lens_index], 2.0 * radius_y[lens_index])
        )
    payload = {
        "model_status": "equivalent first-order model; not a real optical prescription",
        "assumptions": [
            "Paladin Advanced 355-8000",
            "1 mm nominal input diameter",
            "550 urad divergence specification interpreted as full angle, so half angle is 275 urad",
            "M2=1.2 conservative boundary",
            "180 mm geometric centerline path applies to all four cases",
            "1.44 um x 5.2 mm and 1.05 um x 2.667 mm are wafer-plane 1/e^2 full diameters",
            "surface X lies in the incidence plane and is projection-corrected by cos(angle)",
            "laser body length is 90 mm and is used only as a drawing scale reference",
            "equivalent beam-shaping plane is at s=110 mm; near-normal LFC is at s=166 mm; oblique LFC is at s=174 mm",
            "near-normal branch includes an ideal 30 mm objective at s=174 mm after the LFC plane at s=166 mm",
            "objective-to-wafer and oblique-LFC-to-wafer distance is 6 mm",
        ],
        "laser": asdict(laser),
        "performance_requirement": asdict(performance_requirement),
        "performance_results": [asdict(item) for item in performance_results],
        "efficiency_budgets": {
            _case_label(requirement): [asdict(item) for item in default_efficiency_budget(requirement)]
            for requirement in requirements
        },
        "case_path_matrix": path_rows,
        "estimated_layout": [asdict(item) for item in default_estimated_layout()],
        "oblique_75deg_design": {
            "fixed_lfc_focal_x_mm": float(
                np.mean([result.lenses.focal2_x_m for result in oblique_results]) * 1e3
            ),
            "lfc_working_distance_mm": 6.0,
            "maximum_1e2_beam_diameter_x_at_lfc_mm": float(
                max(item[0] for item in beam_diameters_at_lfc) * 1e3
            ),
            "maximum_1e2_beam_diameter_y_at_lfc_mm": float(
                max(item[1] for item in beam_diameters_at_lfc) * 1e3
            ),
            "minimum_recommended_clear_x_mm": float(
                1.5 * max(item[0] for item in beam_diameters_at_lfc) * 1e3
            ),
            "minimum_recommended_clear_y_mm": float(
                1.5 * max(item[1] for item in beam_diameters_at_lfc) * 1e3
            ),
            "practical_clear_aperture_mm": [20.0, 10.0],
            "physical_optics_required": True,
        },
        "results": [
            {
                "requirement": asdict(result.requirement),
                "lenses": asdict(result.lenses),
                "summary": _summary_row(result),
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    figure_path = OUTPUT_DIRECTORY / "four_case_overview.png"
    _plot(results, figure_path)
    topology_path = OUTPUT_DIRECTORY / "four_case_path_topology.png"
    _plot_path_topology(requirements, topology_path)
    performance_figure_path = OUTPUT_DIRECTORY / "performance_compliance.png"
    _plot_performance_compliance(requirements, performance_results, performance_figure_path)

    for row in summary_rows:
        validity = "OK" if row["paraxial_valid_x"] and row["paraxial_valid_y"] else "WARNING"
        print(
            "{case}: angle={incidence_angle_to_normal_deg:.1f} deg, "
            "surface={output_surface_x_um:.6f} um x {output_surface_y_mm:.6f} mm, "
            "fx=({lens1_fx_mm:.6f}, {lens2_fx_mm:.6f}) mm, "
            "fy=({lens1_fy_mm:.6f}, {lens2_fy_mm:.6f}) mm, "
            "paraxial={validity}".format(validity=validity, **row)
        )
    print("Summary: {}".format(summary_path.resolve()))
    print("Performance: {}".format(performance_path.resolve()))
    print("Efficiency:  {}".format(efficiency_budget_path.resolve()))
    print("Cases:   {}".format(path_matrix_path.resolve()))
    print("Layout:  {}".format(layout_path.resolve()))
    print("Results: {}".format(json_path.resolve()))
    print("Figure:  {}".format(figure_path.resolve()))
    print("Paths:   {}".format(topology_path.resolve()))
    print("Compliance: {}".format(performance_figure_path.resolve()))


if __name__ == "__main__":
    main()
