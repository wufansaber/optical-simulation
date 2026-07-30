"""355 nm 高斯光束传播结果绘图和复光场数据导出。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .gaussian import GaussianTracePoint
from .paraxial import OpticalSystem


def _unique_z_points(samples: Sequence[GaussianTracePoint]) -> List[GaussianTracePoint]:
    unique_points = []
    for point in samples:
        if unique_points and abs(unique_points[-1].z - point.z) < 1e-15:
            unique_points[-1] = point
        else:
            unique_points.append(point)
    return unique_points


def calculate_field_map(
    samples: Sequence[GaussianTracePoint],
    transverse_points: int = 401,
    transverse_limit: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 z、横向 y 和复电场 E(y,z)。"""

    if not samples:
        raise ValueError("没有可绘制的高斯光束采样点。")
    if transverse_points < 3:
        raise ValueError("横向采样点数至少为 3。")

    points = _unique_z_points(samples)
    z_positions = np.array([point.z for point in points], dtype=np.float64)
    beam_radii = np.array([point.beam.beam_radius for point in points])
    limit = float(transverse_limit) if transverse_limit is not None else 1.25 * float(beam_radii.max())
    if limit <= 0:
        raise ValueError("横向计算范围必须为正。")

    transverse_positions = np.linspace(-limit, limit, transverse_points)
    field = np.column_stack(
        [point.beam.field(transverse_positions) for point in points]
    )
    return z_positions, transverse_positions, field


def plot_gaussian_propagation(
    samples: Sequence[GaussianTracePoint],
    system: OpticalSystem,
    waist_position: float,
    output_path: Path,
    field_data_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """绘制归一化强度和 1/e^2 光束半径，并保存复电场。"""

    points = _unique_z_points(samples)
    z_positions, transverse_positions, field = calculate_field_map(points)
    intensity = np.array(
        [
            point.beam.intensity(transverse_positions)
            for point in points
        ]
    ).T
    normalized_intensity = intensity / float(intensity.max())
    beam_radii = np.array([point.beam.beam_radius for point in points])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if field_data_path is not None:
        field_data_path = Path(field_data_path)
        field_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            field_data_path,
            z_m=z_positions,
            y_m=transverse_positions,
            electric_field_V_per_m=field,
            intensity_W_per_m2=intensity,
        )

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    color_mesh = axes[0].pcolormesh(
        z_positions,
        transverse_positions * 1e3,
        np.maximum(normalized_intensity, 1e-6),
        shading="auto",
        cmap="inferno",
        norm=LogNorm(vmin=1e-5, vmax=1.0),
    )
    axes[0].plot(z_positions, beam_radii * 1e3, color="cyan", linewidth=1.2, label="+w(z)")
    axes[0].plot(z_positions, -beam_radii * 1e3, color="cyan", linewidth=1.2, label="-w(z)")
    axes[0].axhline(0.0, color="white", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("Transverse y (mm)")
    axes[0].set_title("355 nm coherent Gaussian beam: normalized intensity")
    axes[0].legend(loc="upper right")
    figure.colorbar(color_mesh, ax=axes[0], label="Normalized intensity")

    axes[1].plot(z_positions, beam_radii * 1e6, color="tab:blue", label="Beam radius w(z)")
    axes[1].axvline(waist_position, color="tab:red", linestyle=":", label="Calculated waist")
    axes[1].set_xlabel("Optical-axis position z (m)")
    axes[1].set_ylabel("1/e^2 beam radius (um)")
    axes[1].grid(alpha=0.3)

    for lens_z, lens in system.lens_positions():
        for axis in axes:
            axis.axvline(lens_z, color="tab:green", linestyle="--", alpha=0.75)
        axes[1].text(
            lens_z,
            0.96,
            "Lens f={:g} mm".format(lens.focal_length * 1e3),
            rotation=90,
            va="top",
            ha="right",
            transform=axes[1].get_xaxis_transform(),
            color="tab:green",
        )

    axes[1].legend(loc="best")
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)

