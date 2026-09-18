"""输出双支路晶圆照明系统的完整光路示意图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUTPUT_DIRECTORY = Path("outputs/dual_branch_180mm")

COLORS = {
    "beam": "#6F42C1",
    "common": "#DCEAF7",
    "common_edge": "#4C78A8",
    "switch": "#FCE6BE",
    "switch_edge": "#E59D28",
    "near": "#DCEFD8",
    "near_edge": "#59A14F",
    "oblique": "#F8DEDE",
    "oblique_edge": "#E15759",
    "neutral": "#F1F3F5",
    "neutral_edge": "#555555",
    "text": "#1F2933",
    "wafer": "#202124",
}


def _configure_fonts() -> None:
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def _node(axis, center, width, height, text, group="common", fontsize=8.4, subtitle=None):
    x, y = center
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=COLORS[group],
        edgecolor=COLORS[group + "_edge"],
        linewidth=1.5,
        zorder=4,
    )
    axis.add_patch(box)
    axis.text(x, y + (0.10 if subtitle else 0.0), text, ha="center", va="center", fontsize=fontsize, color=COLORS["text"], zorder=5)
    if subtitle:
        axis.text(x, y - height * 0.30, subtitle, ha="center", va="center", fontsize=6.8, color="#58636D", zorder=5)


def _switch(axis, center, width, height, text, subtitle=None):
    _node(axis, center, width, height, text, group="switch", fontsize=8.4, subtitle=subtitle)


def _beam(axis, start, end, style="-", color=None, linewidth=3.0, zorder=2):
    color = color or COLORS["beam"]
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=linewidth,
            linestyle=style,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def _mirror(axis, center, label, group="neutral", angle=45):
    x, y = center
    length = 0.42
    if angle == 45:
        points = [(x - 0.15, y - 0.21), (x - 0.08, y - 0.28), (x + 0.15, y + 0.21), (x + 0.08, y + 0.28)]
    else:
        points = [(x - 0.15, y + 0.21), (x - 0.08, y + 0.28), (x + 0.15, y - 0.21), (x + 0.08, y - 0.28)]
    axis.add_patch(Polygon(points, closed=True, facecolor=COLORS[group], edgecolor=COLORS[group + "_edge"], linewidth=1.3, zorder=4))
    axis.text(x, y + 0.43, label, ha="center", va="bottom", fontsize=8, color=COLORS["text"], zorder=5)


def _lens(axis, center, label, group="near", subtitle=None):
    x, y = center
    axis.add_patch(
        FancyBboxPatch(
            (x - 0.11, y - 0.42),
            0.22,
            0.84,
            boxstyle="round,pad=0.01,rounding_size=0.10",
            facecolor=COLORS[group],
            edgecolor=COLORS[group + "_edge"],
            linewidth=1.5,
            zorder=4,
        )
    )
    axis.text(x, y + 0.58, label, ha="center", va="bottom", fontsize=8, color=COLORS["text"], zorder=5)
    if subtitle:
        axis.text(x, y - 0.58, subtitle, ha="center", va="top", fontsize=6.7, color="#58636D", zorder=5)


def _draw_common(axis):
    y = 7.45
    axis.add_patch(Rectangle((0.45, 6.55), 10.6, 1.82, facecolor="#F7FAFD", edgecolor=COLORS["common_edge"], linewidth=1.1, zorder=0))
    axis.text(0.65, 8.13, "共用前端（N-A / N-B / O-A / O-B 均通过）", ha="left", va="center", fontsize=11, color=COLORS["common_edge"], weight="bold")
    _beam(axis, (0.85, y), (10.78, y), linewidth=3.2)
    _node(axis, (1.35, y), 1.15, 0.72, "Paladin 355-8000", subtitle="355 nm · >8 W")
    _mirror(axis, (2.25, y), "M0")
    _node(axis, (3.02, y), 0.95, 0.70, "Attenuator", fontsize=7.8)
    _node(axis, (4.00, y), 0.92, 0.70, "Set\nPolarizer", fontsize=7.7)
    _mirror(axis, (4.72, y), "M2", angle=-45)
    _node(axis, (5.55, y), 1.15, 0.72, "Servo D1/D2", fontsize=7.8, subtitle="M0/M2闭环")
    _node(axis, (6.65, y), 1.05, 0.72, "Illumination\nPolarizer", fontsize=7.4)
    _node(axis, (7.62, y), 0.75, 0.70, "Shutter", fontsize=7.6)
    _lens(axis, (8.27, y), "Waist Relay", group="common")
    _node(axis, (9.12, y), 1.05, 0.78, "Y Homogenizer", fontsize=7.5, subtitle="SG36 · U≥90%")
    _mirror(axis, (9.82, y), "M3")
    _mirror(axis, (10.30, y), "M4", angle=-45)
    _switch(axis, (10.86, y), 0.82, 0.82, "FIBE", subtitle="A / B倍率")


def _draw_split_and_branches(axis):
    split_x, split_y = 11.80, 7.45
    diamond = Polygon(
        [(split_x, split_y + 0.48), (split_x + 0.48, split_y), (split_x, split_y - 0.48), (split_x - 0.48, split_y)],
        closed=True,
        facecolor=COLORS["switch"],
        edgecolor=COLORS["switch_edge"],
        linewidth=1.8,
        zorder=5,
    )
    axis.add_patch(diamond)
    axis.text(split_x, split_y, "Prism", ha="center", va="center", fontsize=8.3, color=COLORS["text"], zorder=6)
    axis.text(split_x, split_y + 0.72, "支路切换", ha="center", va="bottom", fontsize=8.2, color=COLORS["switch_edge"], weight="bold")
    _beam(axis, (11.26, split_y), (11.35, split_y), linewidth=3.2)

    # Near-normal path
    near_y = 6.00
    axis.add_patch(Rectangle((11.20, 5.14), 6.10, 1.68, facecolor="#F7FBF6", edgecolor=COLORS["near_edge"], linewidth=1.1, zorder=0))
    axis.text(11.38, 6.58, "近法向专用支路 · 1°", ha="left", va="center", fontsize=10.5, color=COLORS["near_edge"], weight="bold")
    _beam(axis, (split_x, 6.97), (12.15, near_y), color=COLORS["near_edge"], linewidth=2.8)
    _beam(axis, (12.15, near_y), (17.02, near_y), color=COLORS["near_edge"], linewidth=2.8)
    _node(axis, (12.62, near_y), 1.10, 0.72, "Near-normal\nExpander E1", group="near", fontsize=7.5, subtitle="s≈110 mm")
    _mirror(axis, (13.47, near_y), "M5N", group="near")
    _lens(axis, (14.25, near_y), "Normal LFC / E2", group="near", subtitle="s≈166 mm")
    _mirror(axis, (15.05, near_y), "M6N", group="near", angle=-45)
    _node(axis, (16.05, near_y), 1.20, 0.72, "Ideal Objective", group="near", fontsize=7.5, subtitle="f=30 mm · s≈174 mm")

    # Oblique path
    oblique_y = 3.70
    axis.add_patch(Rectangle((11.20, 2.82), 6.10, 1.75, facecolor="#FEF8F8", edgecolor=COLORS["oblique_edge"], linewidth=1.1, zorder=0))
    axis.text(11.38, 4.32, "斜入射专用支路 · 75°", ha="left", va="center", fontsize=10.5, color=COLORS["oblique_edge"], weight="bold")
    _beam(axis, (split_x, 6.95), (12.05, oblique_y), color=COLORS["oblique_edge"], linewidth=2.8)
    _beam(axis, (12.05, oblique_y), (17.02, oblique_y), color=COLORS["oblique_edge"], linewidth=2.8)
    _mirror(axis, (12.36, oblique_y), "M5O", group="oblique")
    _node(axis, (13.16, oblique_y), 1.12, 0.72, "OBE / E1", group="oblique", fontsize=7.8, subtitle="s≈110 mm")
    _mirror(axis, (14.02, oblique_y), "M6O", group="oblique", angle=-45)
    _mirror(axis, (14.74, oblique_y), "M7O", group="oblique")
    _lens(axis, (15.58, oblique_y), "Oblique LFC", group="oblique", subtitle="E2≈174 mm")

    # Wafer targets
    axis.add_patch(Rectangle((17.02, 3.22), 0.16, 3.35, facecolor=COLORS["wafer"], edgecolor=COLORS["wafer"], zorder=5))
    axis.text(17.10, 6.85, "Wafer", ha="center", va="bottom", fontsize=10, color=COLORS["text"], weight="bold")
    axis.text(17.10, 2.98, "s≈180 mm", ha="center", va="top", fontsize=7.2, color="#58636D")
    axis.text(16.98, 6.23, "1°", ha="right", va="center", fontsize=8.5, color=COLORS["near_edge"], weight="bold")
    axis.text(16.98, 3.92, "75°", ha="right", va="center", fontsize=8.5, color=COLORS["oblique_edge"], weight="bold")


def _draw_case_matrix(axis):
    axis.add_patch(Rectangle((0.45, 0.48), 16.85, 1.70, facecolor="#FAFAFA", edgecolor="#AAB2B9", linewidth=1.0, zorder=0))
    axis.text(0.68, 1.95, "四工况切换矩阵", ha="left", va="center", fontsize=10.5, color=COLORS["text"], weight="bold")
    headers = ["工况", "FIBE倍率", "Prism状态", "专用后端", "晶圆面1/e²目标", "性能要求"]
    xs = [1.05, 2.72, 5.05, 8.05, 11.50, 15.25]
    for x, header in zip(xs, headers):
        axis.text(x, 1.64, header, ha="center", va="center", fontsize=7.7, color="#58636D", weight="bold")
    rows = [
        ("N-A", "A", "近法向通过", "Near-normal", "1.44 μm × 5.2 mm", "U≥90% · η≥65%"),
        ("N-B", "B", "近法向通过", "Near-normal", "1.05 μm × 2.667 mm", "U≥90% · η≥65%"),
        ("O-A", "A", "斜入射偏转", "Oblique", "1.44 μm × 5.2 mm", "U≥90% · η≥65%"),
        ("O-B", "B", "斜入射偏转", "Oblique", "1.05 μm × 2.667 mm", "U≥90% · η≥65%"),
    ]
    y_values = [1.35, 1.09, 0.83, 0.57]
    for y, row in zip(y_values, rows):
        for x, value in zip(xs, row):
            axis.text(x, y, value, ha="center", va="center", fontsize=7.4, color=COLORS["text"])


def render() -> None:
    _configure_fonts()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(18, 10))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 9.6)
    axis.axis("off")
    axis.text(0.5, 9.23, "355 nm 双支路晶圆照明完整光路图", ha="left", va="center", fontsize=18, color=COLORS["text"], weight="bold")
    axis.text(17.35, 9.23, "展开几何光程约 180 mm · 示意图非机械比例", ha="right", va="center", fontsize=9.5, color="#58636D")
    axis.text(0.52, 8.83, "公共稳光与匀化 → FIBE倍率切换 → Prism支路切换 → 1°/75°专用整形与聚焦", ha="left", va="center", fontsize=10.5, color="#58636D")
    _draw_common(axis)
    _draw_split_and_branches(axis)
    _draw_case_matrix(axis)
    figure.tight_layout(pad=0.6)
    for suffix in ("png", "svg", "pdf"):
        path = OUTPUT_DIRECTORY / ("complete_optical_path." + suffix)
        figure.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    render()
