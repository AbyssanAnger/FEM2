import torch


class IsotropicMaterial:
    """Isotropes Material für Plane Stress."""

    def __init__(self, E, nu):
        self.E = E  # Young's modulus [Pa]
        self.nu = nu  # Poisson's ratio
        self.C4 = self._compute_stiffness_tensor()

    def _compute_stiffness_tensor(self):
        """Berechnet den Materialtensor C4 für Plane Stress."""
        factor = self.E / (1 - self.nu**2)
        shear_factor = self.E / (2 * (1 + self.nu))

        C4 = torch.zeros(2, 2, 2, 2)
        C4[0, 0, 0, 0] = factor * 1
        C4[1, 1, 1, 1] = factor * 1
        C4[0, 0, 1, 1] = factor * self.nu
        C4[1, 1, 0, 0] = factor * self.nu
        C4[0, 1, 0, 1] = C4[1, 0, 0, 1] = C4[0, 1, 1, 0] = C4[1, 0, 1, 0] = shear_factor
        return C4

    def stress_from_strain(self, eps_2d):
        """Berechnet 2D-Spannung aus 2D-Dehnung (Plane Stress), returns (2x2)."""
        stre_2d = torch.tensordot(self.C4, eps_2d, dims=2)
        return stre_2d
