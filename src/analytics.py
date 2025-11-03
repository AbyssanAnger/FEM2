import torch


def rectangular_section_inertia(width: float, height: float) -> float:
    """Second moment of area I_z for rectangle about z (out-of-plane)."""
    return width * (height**3) / 12.0


def bending_stress_sigma_xx(
    y: torch.Tensor, height: float, width: float, moment: float
) -> torch.Tensor:
    """Sigma_xx(y) = M/I * (y - height/2) for a rectangular section.

    y: tensor of coordinates along the height (0..height)
    height, width: section dims [m]
    moment: bending moment M_z [N*m]
    """
    I_z = rectangular_section_inertia(width, height)
    return (moment / I_z) * (y - height / 2.0)


def cantilever_end_shear_moment_at_x(
    total_end_force: float, length: float, x: float
) -> float:
    """For a cantilever with a point shear force F at the free end (x=length): M(x) = F * (length - x)."""
    return total_end_force * (length - x)


def analytical_sigma_xx_midspan(
    height: float,
    width: float,
    length: float,
    total_end_force: float,
    num_points: int = 100,
):
    """Returns (y, sigma_xx) along the height at x = L/2 for a cantilever with end force F.

    total_end_force: resultant force at free end [N] (positive sign will be carried through)
    """
    x_eval = length / 2.0
    M = cantilever_end_shear_moment_at_x(total_end_force, length, x_eval)
    y = torch.linspace(0.0, height, num_points, dtype=torch.get_default_dtype())
    sigma = bending_stress_sigma_xx(y, height, width, M)
    return y, sigma
