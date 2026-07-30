"""直接使用 API 计算 355 nm 光束经过单薄透镜后的束腰。"""

from optical_sim import FreeSpace, GaussianBeam, OpticalSystem, ThinLens, sample_gaussian_system


def main() -> None:
    beam = GaussianBeam.from_waist(
        wavelength=355e-9,
        waist_radius=0.5e-3,
        waist_position=0.0,
        power=1.0,
    )
    system = OpticalSystem([FreeSpace(0.20), ThinLens(0.10), FreeSpace(0.16)])
    output = sample_gaussian_system(system, beam, samples_per_space=100)[-1]

    waist_z = output.beam.waist_position(output.z)
    print("束腰全局位置: {:.9f} m".format(waist_z))
    print("束腰半径: {:.6f} um".format(output.beam.waist_radius * 1e6))
    print("束腰直径: {:.6f} um".format(2.0 * output.beam.waist_radius * 1e6))


if __name__ == "__main__":
    main()

