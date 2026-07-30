"""求解默认 355 nm 条形光斑处方并打印镜片参数。"""

from optical_sim import StripeDesignConfig, solve_stripe_design


def main():
    design = solve_stripe_design(StripeDesignConfig())
    for lens in design.prescription:
        print(lens)
    print("Surface X diameter: {:.9f} um".format(design.surface_diameter_x * 1e6))
    print("Surface Y diameter: {:.9f} mm".format(design.surface_diameter_y * 1e3))


if __name__ == "__main__":
    main()

