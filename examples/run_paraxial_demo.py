"""最小使用示例：建立系统、追迹光线并输出图像。"""

from pathlib import Path

from optical_sim import FreeSpace, OpticalSystem, Ray, ThinLens
from optical_sim.plotting import plot_ray_traces


def main() -> None:
    system = OpticalSystem(
        [
            FreeSpace(distance=0.10),
            ThinLens(focal_length=0.08),
            FreeSpace(distance=0.08),
        ]
    )
    input_rays = [Ray(height=height_mm / 1_000, angle=0.0) for height_mm in (-10, -5, 0, 5, 10)]

    for ray in input_rays:
        output_ray = system.propagate(ray)
        print(f"输入 {ray} -> 输出 {output_ray}")

    plot_ray_traces(system, input_rays, Path("outputs/example.png"))


if __name__ == "__main__":
    main()k

