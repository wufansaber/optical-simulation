"""逐透镜复光场输出与 X/Y 光路包络示意图。"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from math import cos, pi, radians, sin
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .astigmatic import AstigmaticGaussianBeam
from .gaussian import SPEED_OF_LIGHT, VACUUM_PERMITTIVITY
from .stripe_model import LensPrescription, StripeDesign


@dataclass(frozen=True)
class LensPlaneState:
    """光束刚通过理想薄透镜后的状态。"""

    lens: LensPrescription
    beam_after: AstigmaticGaussianBeam
    surface_angle_to_normal_deg: float = 0.0


@dataclass(frozen=True)
class OpticalPathSample:
    z: float
    radius_x: float
    radius_y: float


def _input_beam(design: StripeDesign) -> AstigmaticGaussianBeam:
    config = design.config
    return AstigmaticGaussianBeam.from_plane(
        wavelength=config.wavelength,
        radius_x=config.input_diameter / 2.0,
        radius_y=config.input_diameter / 2.0,
        power=config.power,
    )


def lens_plane_states(design: StripeDesign) -> List[LensPlaneState]:
    """返回各光学元件面后的复光束状态。"""

    beam = _input_beam(design)
    states = []
    for lens in design.prescription:
        beam = beam.apply_lens(focal_x=lens.focal_x, focal_y=lens.focal_y)
        states.append(
            LensPlaneState(
                lens=lens,
                beam_after=beam,
                surface_angle_to_normal_deg=0.0,
            )
        )
        beam = beam.free_space(lens.distance_to_next)
    return states


def sample_optical_path(
    design: StripeDesign,
    samples_per_segment: int = 80,
) -> List[OpticalPathSample]:
    """沿每段自由空间采样真实 X/Y 1/e² 半径。"""

    if samples_per_segment < 2:
        raise ValueError("每段光路至少需要两个采样点。")
    beam = _input_beam(design)
    samples = [OpticalPathSample(0.0, beam.radius_x, beam.radius_y)]
    for lens in design.prescription:
        beam = beam.apply_lens(focal_x=lens.focal_x, focal_y=lens.focal_y)
        samples.append(OpticalPathSample(lens.z, beam.radius_x, beam.radius_y))
        start_beam = beam
        distances = np.linspace(0.0, lens.distance_to_next, samples_per_segment + 1)[1:]
        for distance in distances:
            propagated = start_beam.free_space(float(distance))
            samples.append(
                OpticalPathSample(
                    lens.z + float(distance),
                    propagated.radius_x,
                    propagated.radius_y,
                )
            )
        beam = start_beam.free_space(lens.distance_to_next)
    return samples


def _safe_file_stem(lens: LensPrescription) -> str:
    clean_name = re.sub(r"[^A-Za-z0-9_-]+", "_", lens.name).strip("_")
    return "{:02d}_{}_after".format(lens.index, clean_name)


def _curvature(q_value: complex) -> float:
    inverse_curvature = (1.0 / q_value).real
    if abs(inverse_curvature) < 1e-15:
        return float("inf")
    return 1.0 / inverse_curvature


def _field_grid(state: LensPlaneState, grid_points: int):
    beam = state.beam_after
    x_coordinates = np.linspace(
        -1.25 * state.lens.surface_footprint_x,
        1.25 * state.lens.surface_footprint_x,
        grid_points,
    )
    y_coordinates = np.linspace(
        -1.25 * state.lens.surface_footprint_y,
        1.25 * state.lens.surface_footprint_y,
        grid_points,
    )
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    axial_offset = None
    if state.surface_angle_to_normal_deg == 0.0:
        electric_field = beam.field(x_grid, y_grid)
    else:
        angle = radians(state.surface_angle_to_normal_deg)
        axial_offset = x_grid * sin(angle)
        transverse_x = x_grid * cos(angle)
        qx = beam.qx + axial_offset
        qy = beam.qy + axial_offset
        diffraction_scale = np.sqrt(
            (1.0 + axial_offset / beam.qx) * (1.0 + axial_offset / beam.qy)
        )
        axis_field = (
            beam.axis_field
            * np.exp(1j * beam.wave_number * axial_offset)
            / diffraction_scale
        )
        transverse_phase = -1j * beam.wave_number * (
            transverse_x ** 2 / (2.0 * qx) + y_grid ** 2 / (2.0 * qy)
        )
        electric_field = axis_field * np.exp(transverse_phase)
    intensity = (
        0.5
        * beam.refractive_index
        * VACUUM_PERMITTIVITY
        * SPEED_OF_LIGHT
        * np.abs(electric_field) ** 2
    )
    normalized = intensity / float(intensity.max())
    center = grid_points // 2
    phase_field = electric_field
    if axial_offset is not None:
        phase_field = electric_field * np.exp(-1j * beam.wave_number * axial_offset)
    relative_phase = np.angle(
        phase_field * np.exp(-1j * np.angle(phase_field[center, center]))
    )
    return x_coordinates, y_coordinates, electric_field, intensity, normalized, relative_phase


def _plot_lens_plane(
    state: LensPlaneState,
    image_path: Path,
    grid_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, field, intensity, normalized, phase = _field_grid(state, grid_points)
    beam = state.beam_after
    coordinate_prefix = "Element surface" if state.surface_angle_to_normal_deg else "Local"
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    intensity_mesh = axes[0].pcolormesh(
        x * 1e3,
        y * 1e3,
        normalized,
        shading="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].contour(
        x * 1e3,
        y * 1e3,
        normalized,
        levels=[np.exp(-2.0)],
        colors=["#d97706"],
        linewidths=1.4,
    )
    axes[0].set(
        title="Normalized intensity",
        xlabel=coordinate_prefix + " X (mm)",
        ylabel=coordinate_prefix + " Y (mm)",
    )
    axes[0].text(
        0.02,
        0.02,
        "Orange: 1/e^2 contour\nAxes independently scaled",
        transform=axes[0].transAxes,
        va="bottom",
        color="#202124",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    figure.colorbar(intensity_mesh, ax=axes[0], label="Normalized intensity")

    phase_mesh = axes[1].pcolormesh(
        x * 1e3,
        y * 1e3,
        phase,
        shading="auto",
        cmap="twilight",
        vmin=-pi,
        vmax=pi,
    )
    phase_title = (
        "Phase after removing oblique carrier"
        if state.surface_angle_to_normal_deg
        else "Phase relative to optical axis"
    )
    axes[1].set(
        title=phase_title,
        xlabel=coordinate_prefix + " X (mm)",
        ylabel=coordinate_prefix + " Y (mm)",
    )
    figure.colorbar(phase_mesh, ax=axes[1], label="Wrapped phase (rad)")
    if state.surface_angle_to_normal_deg:
        axes[1].text(
            0.02,
            0.02,
            "Plot removes the carrier tilt; NPZ retains sampled total complex field",
            transform=axes[1].transAxes,
            va="bottom",
            color="#202124",
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
        )

    figure.suptitle(
        "{} after lens | z={:.6f} mm | 1/e^2 diameters: X={:.6f} mm, Y={:.6f} mm".format(
            state.lens.name,
            state.lens.z * 1e3,
            state.lens.surface_footprint_x * 1e3,
            state.lens.surface_footprint_y * 1e3,
        ),
        color="#202124",
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image_path, dpi=320, facecolor="white")
    plt.close(figure)
    return x, y, field, intensity


def _plot_lens_overview(
    states: Sequence[LensPlaneState],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.5), constrained_layout=True)
    coordinate = np.linspace(-2.5, 2.5, 181)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    normalized = np.exp(-2.0 * (x_grid ** 2 + y_grid ** 2))

    for state, axis in zip(states, axes.flat):
        axis.pcolormesh(
            coordinate,
            coordinate,
            normalized,
            shading="auto",
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
        )
        axis.contour(
            coordinate,
            coordinate,
            normalized,
            levels=[np.exp(-2.0)],
            colors=["#d97706"],
            linewidths=1.0,
        )
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            "{} | z={:.3f} mm\nDx={:.3f} mm, Dy={:.3f} mm".format(
                state.lens.name,
                state.lens.z * 1e3,
                state.lens.surface_footprint_x * 1e3,
                state.lens.surface_footprint_y * 1e3,
            ),
            fontsize=10,
        )

    for unused_axis in list(axes.flat)[len(states) :]:
        unused_axis.axis("off")
        unused_axis.text(
            0.05,
            0.58,
            "Each panel uses normalized X/wx and Y/wy coordinates.\n\n"
            "Physical 1/e^2 diameters are listed in each title.\n"
            "Individual PNG/NPZ files use physical millimetre axes.",
            transform=unused_axis.transAxes,
            va="center",
            color="#202124",
        )
    figure.suptitle(
        "355 nm field overview immediately after each lens\n"
        "Normalized coordinates; orange contour is 1/e^2",
        color="#202124",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=320, facecolor="white")
    plt.close(figure)


def _plot_optical_layout(
    design: StripeDesign,
    output_path: Path,
) -> None:
    samples = sample_optical_path(design)
    z_mm = np.array([sample.z for sample in samples]) * 1e3
    x_mm = np.array([sample.radius_x for sample in samples]) * 1e3
    y_mm = np.array([sample.radius_y for sample in samples]) * 1e3

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    envelope_axes = axes
    axes[0].fill_between(z_mm, -x_mm, x_mm, color="#90b7dc", alpha=0.55)
    axes[0].plot(z_mm, x_mm, color="#1f5a94", linewidth=1.8, label="+wx")
    axes[0].plot(z_mm, -x_mm, color="#1f5a94", linewidth=1.8, linestyle="--", label="-wx")
    axes[0].set_ylabel("X 1/e^2 radius (mm)")
    axes[0].set_title("X envelope: expansion and cylindrical focus")

    axes[1].fill_between(z_mm, -y_mm, y_mm, color="#efd38b", alpha=0.6)
    axes[1].plot(z_mm, y_mm, color="#a16207", linewidth=1.8, label="+wy")
    axes[1].plot(z_mm, -y_mm, color="#a16207", linewidth=1.8, linestyle="--", label="-wy")
    axes[1].set_ylabel("Y 1/e^2 radius (mm)")
    axes[1].set_xlabel("Optical-axis position z (mm)")
    axes[1].set_title("Y envelope: expanded and approximately collimated at the target")

    for lens in design.prescription:
        if lens.focal_x is not None and lens.focal_y is not None:
            color, line_style = "#374151", "-"
        elif lens.focal_x is not None:
            color, line_style = "#1f5a94", "-"
        else:
            color, line_style = "#a16207", "--"
        for axis in envelope_axes:
            axis.axvline(lens.z * 1e3, color=color, linestyle=line_style, alpha=0.8, linewidth=1.1)
        label_height = 0.92 if lens.index % 2 else 0.72
        axes[0].text(
            lens.z * 1e3,
            label_height,
            lens.name,
            rotation=90,
            transform=axes[0].get_xaxis_transform(),
            ha="right",
            va="top",
            color=color,
            fontsize=9,
        )

    for axis in envelope_axes:
        axis.axvline(design.surface_z * 1e3, color="#202124", linestyle=":", linewidth=1.6)
        axis.grid(color="#d1d5db", alpha=0.45)
        axis.axhline(0.0, color="#6b7280", linewidth=0.7)
    axes[0].text(
        design.surface_z * 1e3,
        0.98,
        "target: normal incidence",
        rotation=90,
        transform=axes[0].get_xaxis_transform(),
        ha="right",
        va="top",
        color="#202124",
    )

    legend_handles = [
        Line2D([0], [0], color="#1f5a94", label="X cylindrical lens"),
        Line2D([0], [0], color="#a16207", linestyle="--", label="Y cylindrical lens"),
        Line2D([0], [0], color="#374151", label="Spherical lens"),
        Line2D([0], [0], color="#202124", linestyle=":", label="Target plane"),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.92,
    )

    axes[0].annotate(
        "Laser output\nDx=Dy={:.3f} mm".format(design.config.input_diameter * 1e3),
        xy=(0.0, design.config.input_diameter * 0.5e3),
        xytext=(18.0, 18.0),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#4b5563", "linewidth": 0.9},
        color="#374151",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    axes[1].text(
        0.985,
        0.965,
        "Target 1/e^2 diameters\nX={:.6f} um, Y={:.6f} mm".format(
            design.surface_diameter_x * 1e6,
            design.surface_diameter_y * 1e3,
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )
    figure.suptitle(
        "355 nm complete optical path: laser output to normal-incidence focus\n"
        "Thin-lens ABCD model; envelopes show calculated X/Y 1/e^2 radii",
        color="#202124",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=320, facecolor="white")
    plt.close(figure)

def export_lens_plane_outputs(
    design: StripeDesign,
    output_directory: Path,
    overview_path: Path,
    layout_path: Path,
    grid_points: int = 401,
) -> None:
    """输出七个透镜面 PNG/NPZ、总览、清单及光路示意图。"""

    if grid_points < 101 or grid_points % 2 == 0:
        raise ValueError("透镜面网格点数必须是大于等于 101 的奇数。")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    states = lens_plane_states(design)
    manifest_rows = []

    for state in states:
        stem = _safe_file_stem(state.lens)
        image_path = output_directory / (stem + ".png")
        data_path = output_directory / (stem + ".npz")
        x, y, field, intensity = _plot_lens_plane(state, image_path, grid_points)
        beam = state.beam_after
        np.savez_compressed(
            data_path,
            x_m=x,
            y_m=y,
            electric_field_V_per_m=field,
            intensity_W_per_m2=intensity,
            z_m=state.lens.z,
            wavelength_m=beam.wavelength,
            diameter_x_m=state.lens.surface_footprint_x,
            diameter_y_m=state.lens.surface_footprint_y,
            beam_normal_diameter_x_m=2.0 * beam.radius_x,
            beam_normal_diameter_y_m=2.0 * beam.radius_y,
            element_surface_angle_to_normal_deg=state.surface_angle_to_normal_deg,
            curvature_x_m=_curvature(beam.qx),
            curvature_y_m=_curvature(beam.qy),
            qx=beam.qx,
            qy=beam.qy,
        )
        manifest_rows.append(
            [
                state.lens.index,
                state.lens.name,
                "{:.6f}".format(state.lens.z * 1e3),
                "{:.6f}".format(state.lens.surface_footprint_x * 1e3),
                "{:.6f}".format(state.lens.surface_footprint_y * 1e3),
                image_path.name,
                data_path.name,
            ]
        )

    manifest_path = output_directory / "lens_planes_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "index",
                "name",
                "s_mm",
                "element_surface_1e2_footprint_x_mm",
                "element_surface_1e2_footprint_y_mm",
                "image",
                "field_data",
            ]
        )
        writer.writerows(manifest_rows)

    _plot_lens_overview(states, Path(overview_path))
    _plot_optical_layout(design, Path(layout_path))
