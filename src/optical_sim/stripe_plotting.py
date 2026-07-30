"""工件面条形光斑的二维光场、截面和数据导出。"""

from __future__ import annotations

from math import exp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .stripe_model import StripeDesign, surface_field, surface_intensity, write_prescription_csv


def export_surface_field(
    design: StripeDesign,
    figure_path: Path,
    field_data_path: Path,
    prescription_path: Path,
    grid_points: int = 601,
    show: bool = False,
) -> None:
    """导出二维复光场 NPZ、镜片 CSV 和带中心截面的静态图。"""

    if grid_points < 101:
        raise ValueError("二维光场每个方向至少需要 101 个采样点。")
    x_limit = 1.25 * design.config.target_surface_diameter_x
    y_limit = 1.25 * design.config.target_surface_diameter_y
    x_surface = np.linspace(-x_limit, x_limit, grid_points)
    y_surface = np.linspace(-y_limit, y_limit, grid_points)
    x_grid, y_grid = np.meshgrid(x_surface, y_surface)
    electric_field = surface_field(design, x_grid, y_grid)
    intensity = surface_intensity(design, x_grid, y_grid)
    normalized = intensity / float(intensity.max())

    figure_path = Path(figure_path)
    field_data_path = Path(field_data_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    field_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        field_data_path,
        x_surface_m=x_surface,
        y_surface_m=y_surface,
        x_surface_um=x_surface * 1e6,
        y_surface_um=y_surface * 1e6,
        electric_field_V_per_m=electric_field,
        intensity_W_per_m2=intensity,
        normalized_intensity=normalized,
        wavelength_m=design.config.wavelength,
        incidence_angle_to_normal_deg=design.config.incidence_angle_to_normal_deg,
        incidence_angle_to_plane_deg=design.incidence_angle_to_plane_deg,
        paraxial_slope_x=design.paraxial_slope_x,
        geometric_na_x=design.geometric_na_x,
        paraxial_model_valid_x=design.paraxial_model_valid_x,
        measured_1e2_diameter_x_m=design.surface_diameter_x,
        measured_1e2_diameter_y_m=design.surface_diameter_y,
        measured_1e2_diameter_x_um=design.surface_diameter_x * 1e6,
        measured_1e2_diameter_y_um=design.surface_diameter_y * 1e6,
    )
    write_prescription_csv(design, prescription_path)

    center_x_index = int(np.argmin(np.abs(x_surface)))
    center_y_index = int(np.argmin(np.abs(y_surface)))
    x_profile = normalized[center_y_index, :]
    y_profile = normalized[:, center_x_index]

    figure = plt.figure(figsize=(12.5, 7.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0))
    field_axis = figure.add_subplot(grid[:, 0])
    x_axis = figure.add_subplot(grid[0, 1])
    y_axis = figure.add_subplot(grid[1, 1])

    color_mesh = field_axis.pcolormesh(
        x_surface * 1e6,
        y_surface * 1e6,
        normalized,
        shading="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
    )
    field_axis.contour(
        x_surface * 1e6,
        y_surface * 1e6,
        normalized,
        levels=[exp(-2.0)],
        colors=["#d97706"],
        linewidths=3.0,
    )
    field_axis.set_title("Surface intensity map (equal physical scale)")
    field_axis.set_xlabel("Surface X (um)")
    field_axis.set_ylabel("Surface Y (um)")
    field_axis.set_aspect("equal", adjustable="box")
    field_axis.text(
        0.02,
        0.02,
        "Orange: 1/e^2 contour\nBoth axes in um with identical physical scale\n"
        "The 0.8 um X width is below one output pixel",
        transform=field_axis.transAxes,
        va="bottom",
        color="#202124",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    if not design.paraxial_model_valid_x:
        field_axis.text(
            0.02,
            0.98,
            "ABCD extrapolation only: paraxial slope = {:.3f} > 0.3".format(
                design.paraxial_slope_x
            ),
            transform=field_axis.transAxes,
            va="top",
            color="#991b1b",
            bbox={"facecolor": "#fef2f2", "edgecolor": "#991b1b", "alpha": 0.95},
        )
    figure.colorbar(color_mesh, ax=field_axis, label="Normalized intensity")

    threshold = exp(-2.0)
    x_axis.plot(x_surface * 1e6, x_profile, color="#1f5a94", linewidth=2.0)
    x_axis.axhline(threshold, color="#202124", linestyle=":", label="1/e^2")
    x_axis.axvline(design.surface_x_negative * 1e6, color="#d97706", linestyle="--")
    x_axis.axvline(design.surface_x_positive * 1e6, color="#d97706", linestyle="--")
    x_axis.set(
        title="X center profile: {:.6f} um".format(design.surface_diameter_x * 1e6),
        xlabel="Surface X (um)",
        ylabel="Normalized intensity",
        ylim=(-0.02, 1.04),
    )
    x_axis.grid(alpha=0.2)
    x_axis.legend(loc="upper right")

    y_radius = design.surface_diameter_y / 2.0
    y_axis.plot(y_surface * 1e6, y_profile, color="#1f5a94", linewidth=2.0)
    y_axis.axhline(threshold, color="#202124", linestyle=":", label="1/e^2")
    y_axis.axvline(-y_radius * 1e6, color="#d97706", linestyle="--")
    y_axis.axvline(y_radius * 1e6, color="#d97706", linestyle="--")
    y_axis.set(
        title="Y center profile: {:.6f} um".format(design.surface_diameter_y * 1e6),
        xlabel="Surface Y (um)",
        ylabel="Normalized intensity",
        ylim=(-0.02, 1.04),
    )
    y_axis.grid(alpha=0.2)
    y_axis.legend(loc="upper right")

    figure.suptitle(
        "355 nm astigmatic Gaussian field on the target surface\n"
        "Beam-to-normal = {:.1f} deg (beam-to-plane = {:.1f} deg); "
        "1/e^2 full diameters; incidence plane X-Z".format(
            design.config.incidence_angle_to_normal_deg,
            design.incidence_angle_to_plane_deg,
        ),
        color="#202124",
    )
    figure.savefig(figure_path, dpi=320, facecolor="white")
    if show:
        plt.show()
    plt.close(figure)
