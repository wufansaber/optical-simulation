"""355 nm 相干像散高斯光束与近轴光学工具包。"""

from .astigmatic import AstigmaticGaussianBeam
from .gaussian import (
    GaussianBeam,
    GaussianTracePoint,
    propagate_gaussian_system,
    sample_gaussian_system,
)
from .paraxial import FreeSpace, OpticalSystem, Ray, ThinLens, TracePoint
from .interactive_model import (
    InteractiveConfig,
    InteractiveLens,
    InteractiveResult,
    OutputPlaneIntensity,
    PropagationDirection,
    intensity_on_output_z_plane,
    sample_interactive_path,
    simulate_to_z,
    stripe_355nm_preset,
)
from .plane_outputs import (
    LensPlaneState,
    export_lens_plane_outputs,
    lens_plane_states,
    sample_optical_path,
)
from .stripe_model import (
    LensPrescription,
    StripeDesign,
    StripeDesignConfig,
    solve_stripe_design,
    surface_field,
    surface_intensity,
)

__all__ = [
    "AstigmaticGaussianBeam",
    "FreeSpace",
    "GaussianBeam",
    "GaussianTracePoint",
    "InteractiveConfig",
    "InteractiveLens",
    "InteractiveResult",
    "LensPrescription",
    "LensPlaneState",
    "OpticalSystem",
    "OutputPlaneIntensity",
    "PropagationDirection",
    "Ray",
    "StripeDesign",
    "StripeDesignConfig",
    "ThinLens",
    "TracePoint",
    "propagate_gaussian_system",
    "intensity_on_output_z_plane",
    "export_lens_plane_outputs",
    "lens_plane_states",
    "sample_gaussian_system",
    "sample_interactive_path",
    "sample_optical_path",
    "solve_stripe_design",
    "simulate_to_z",
    "stripe_355nm_preset",
    "surface_field",
    "surface_intensity",
]
__version__ = "0.7.0"
