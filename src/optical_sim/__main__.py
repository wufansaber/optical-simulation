"""355 nm X/Y 独立扩束和倾斜平面条形光斑设计入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .plane_outputs import export_lens_plane_outputs
from .stripe_model import StripeDesignConfig, solve_stripe_design
from .stripe_plotting import export_surface_field


def _format_focal(focal):
    return "--" if focal is None else "{:+.6f}".format(focal * 1e3)


def run_design(args: argparse.Namespace) -> None:
    config = StripeDesignConfig(
        wavelength=args.wavelength_nm * 1e-9,
        input_diameter=args.input_diameter_mm * 1e-3,
        target_surface_diameter_x=args.target_x_um * 1e-6,
        target_surface_diameter_y=args.target_y_mm * 1e-3,
        incidence_angle_to_normal_deg=args.angle_to_normal_deg,
        power=args.power_w,
    )
    design = solve_stripe_design(config)
    export_surface_field(
        design=design,
        figure_path=args.output,
        field_data_path=args.field_data,
        prescription_path=args.prescription,
        grid_points=args.grid_points,
        show=args.show,
    )
    export_lens_plane_outputs(
        design=design,
        output_directory=args.lens_plane_dir,
        overview_path=args.lens_overview,
        layout_path=args.layout_output,
        grid_points=args.lens_grid_points,
    )

    print("=== 355 nm X/Y 独立高斯光束设计 ===")
    print("尺寸定义: 1/e^2 全直径；入射面: X-Z；柱面镜轴: Y")
    print("输入光斑: {:.6f} mm".format(config.input_diameter * 1e3))
    print("光束与表面法线夹角: {:.3f} deg".format(config.incidence_angle_to_normal_deg))
    print("光束与表面夹角:     {:.3f} deg".format(design.incidence_angle_to_plane_deg))
    print()
    print("序号  元件  z/mm        fx/mm        fy/mm        到下一面/mm")
    for lens in design.prescription:
        print(
            "{:>2d}    {:<3s}  {:>10.6f}  {:>11s}  {:>11s}  {:>12.6f}".format(
                lens.index,
                lens.name,
                lens.z * 1e3,
                _format_focal(lens.focal_x),
                _format_focal(lens.focal_y),
                lens.distance_to_next * 1e3,
            )
        )
    print()
    print("靶面累积光程坐标:      {:.6f} mm".format(design.surface_z * 1e3))
    print("X 表面 1/e^2 全直径:  {:.9f} um".format(design.surface_diameter_x * 1e6))
    print("Y 表面 1/e^2 全直径:  {:.9f} um".format(design.surface_diameter_y * 1e6))
    print("X 束腰半径(垂直截面): {:.9f} um".format(design.beam_at_surface.waist_radius_x * 1e6))
    print("X 近轴斜率参数:       {:.6f}".format(design.paraxial_slope_x))
    print("X 等效几何 NA:         {:.6f}".format(design.geometric_na_x))
    print("X 等效锥角半角:       {:.6f} deg".format(design.equivalent_cone_half_angle_deg))
    print("峰值强度(功率 {:.3f} W): {:.6e} W/m^2".format(config.power, design.beam_at_surface.peak_intensity))
    print("二维光场图:            {}".format(args.output.resolve()))
    print("复电场数据:            {}".format(args.field_data.resolve()))
    print("光学元件处方:          {}".format(args.prescription.resolve()))
    print("逐元件光场目录:        {}".format(args.lens_plane_dir.resolve()))
    print("元件面光场总览:        {}".format(args.lens_overview.resolve()))
    print("光路示意图:            {}".format(args.layout_output.resolve()))
    if not design.paraxial_model_valid_x:
        print(
            "警告: 近轴斜率 {:.3f} > 0.3，该处方是 ABCD/高斯模型的外推结果，"
            "不能作为可直接加工的物理处方。".format(design.paraxial_slope_x)
        )
        print("工程定型前必须使用高 NA 矢量衍射、实际物镜及像差模型重新优化。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="355 nm 倾斜平面条形光斑设计。")
    parser.add_argument("--wavelength-nm", type=float, default=355.0)
    parser.add_argument("--input-diameter-mm", type=float, default=1.2)
    parser.add_argument("--target-x-um", type=float, default=0.8)
    parser.add_argument("--target-y-mm", type=float, default=1.5)
    parser.add_argument(
        "--angle-to-normal-deg",
        "--angle-deg",
        dest="angle_to_normal_deg",
        type=float,
        default=0.0,
        help="光束与工件表面法线的夹角。",
    )
    parser.add_argument("--power-w", type=float, default=1.0)
    parser.add_argument("--grid-points", type=int, default=601)
    parser.add_argument("--output", type=Path, default=Path("outputs/stripe_355nm_surface.png"))
    parser.add_argument("--field-data", type=Path, default=Path("outputs/stripe_355nm_field.npz"))
    parser.add_argument(
        "--prescription",
        type=Path,
        default=Path("outputs/stripe_355nm_lens_prescription.csv"),
    )
    parser.add_argument(
        "--lens-plane-dir",
        type=Path,
        default=Path("outputs/lens_planes"),
        help="逐透镜面 PNG/NPZ 输出目录。",
    )
    parser.add_argument(
        "--lens-overview",
        type=Path,
        default=Path("outputs/stripe_355nm_lens_planes_overview.png"),
    )
    parser.add_argument(
        "--layout-output",
        type=Path,
        default=Path("outputs/stripe_355nm_optical_layout.png"),
    )
    parser.add_argument("--lens-grid-points", type=int, default=401)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_design(parse_args())


if __name__ == "__main__":
    main()
