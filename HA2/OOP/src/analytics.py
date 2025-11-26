import torch


def rectangular_section_inertia(width: float, height: float) -> float:
    """Second moment of area I_z for rectangle about z (out-of-plane)."""
    return width * (height**3) / 12.0


def analytical_sigma_xx_midspan(
    height: float,
    width: float,
    length: float,
    force: float,
    mid_point_x: float,
    num_points: int = 100,
):
    """Returns (y, sigma_xx) along the height at x = mid_point_x.

    Uses exact same formula as original: M_z = 1000.0 * (length - mid_point_x)
    The force should be the magnitude of the resultant force (positive value).

    height, width: section dimensions [m]
    length: beam length [m]
    force: magnitude of resultant force at free end [N] (will use absolute value)
    mid_point_x: x-coordinate where to evaluate [m]
    """
    I_z = (width * height**3) / 12.0
    M_z = abs(force) * (
        length - mid_point_x
    )  # F * (L-x), use absolute value as in original
    y_analytical = torch.linspace(
        0.0, height, num_points, dtype=torch.get_default_dtype()
    )
    sigma_analytical = (
        M_z / I_z * (y_analytical - height / 2.0)
    ) / 1e6  # in MPa, as in original
    return y_analytical, sigma_analytical
