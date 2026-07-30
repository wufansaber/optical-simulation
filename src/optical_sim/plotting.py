"""近轴光线追迹绘图（保留用于几何光线对照）。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt

from .paraxial import OpticalSystem, Ray


def plot_ray_traces(
    system: OpticalSystem,
    rays: Sequence[Ray],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    if not rays:
        raise ValueError("至少需要一条输入光线。")

    figure, axes = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for index, ray in enumerate(rays, start=1):
        trace = system.trace(ray)
        axes.plot(
            [point.z for point in trace],
            [point.ray.height * 1e3 for point in trace],
            marker=".",
            linewidth=1.5,
            label="Ray {}".format(index),
        )
    for z, lens in system.lens_positions():
        axes.axvline(z, color="tab:blue", linestyle="--", alpha=0.5)
        axes.text(z, 0.98, "f={:g} mm".format(lens.focal_length * 1e3), rotation=90,
                  va="top", ha="right", transform=axes.get_xaxis_transform())
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set(title="Paraxial ray trace", xlabel="z (m)", ylabel="y (mm)")
    axes.grid(alpha=0.25)
    axes.legend(loc="best")
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=160)
    if show:
        plt.show()
    plt.close(figure)

