import torch


class IsotropicMaterial:
    """Isotropes lineares Elastizitätsmaterial (2D plane stress)."""

    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))
        self.lame = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        self.C4 = self._build_stiffness_tensor()

    def _build_stiffness_tensor(self):
        tdm = 2
        C4 = torch.zeros(tdm, tdm, tdm, tdm, dtype=torch.float64)
        C4[0, 0, 0, 0] = 1
        C4[1, 1, 1, 1] = 1
        C4[1, 1, 0, 0] = self.nu
        C4[0, 0, 1, 1] = self.nu
        C4[0, 1, 1, 0] = (1 - self.nu) / 2
        C4[1, 0, 1, 0] = (1 - self.nu) / 2
        C4[0, 1, 0, 1] = (1 - self.nu) / 2
        C4[1, 0, 0, 1] = (1 - self.nu) / 2
        C4 *= self.E / (1 - self.nu**2)
        return C4

    def stress_from_strain(self, eps, ei):
        """Berechnet Spannung σ = C : ε."""
        trace_eps = torch.trace(eps)
        return 2 * self.mu * eps + self.lame * trace_eps * ei
