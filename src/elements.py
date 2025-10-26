import torch
import math


def gauss_quadrature(nqp=4, ndm=2):
    """Definiert Gauss-Punkte und -Gewichte für 2x2-Integration."""
    a = math.sqrt(3) / 3
    qpt = torch.tensor([[-a, -a], [a, -a], [-a, a], [a, a]], dtype=torch.float64)
    w8 = torch.ones(nqp, 1, dtype=torch.float64)
    return qpt, w8


def master_element(qpt, nen=4):
    """Berechnet Shape-Funktionen N und Ableitungen gamma im Master-Element."""
    nqp = qpt.shape[0]
    N = torch.zeros(nqp, nen, dtype=torch.float64)
    gamma = torch.zeros(nqp, nen, 2, dtype=torch.float64)  # Explizit 2 für ndm
    for q in range(nqp):
        xi = qpt[q]
        # Shape-Funktionen (bilinear Q4)
        N[q, 0] = 0.25 * (1 - xi[0]) * (1 - xi[1])  # SW
        N[q, 1] = 0.25 * (1 + xi[0]) * (1 - xi[1])  # SE
        N[q, 2] = 0.25 * (1 + xi[0]) * (1 + xi[1])  # NE
        N[q, 3] = 0.25 * (1 - xi[0]) * (1 + xi[1])  # NW

        # Ableitungen gamma (∂N/∂ξ, ∂N/∂η) – VOLLSTÄNDIG!
        # Knoten 0 (SW)
        gamma[q, 0, 0] = -0.25 * (1 - xi[1])  # ∂N0/∂ξ
        gamma[q, 0, 1] = -0.25 * (1 - xi[0])  # ∂N0/∂η
        # Knoten 1 (SE)
        gamma[q, 1, 0] = 0.25 * (1 - xi[1])  # ∂N1/∂ξ
        gamma[q, 1, 1] = -0.25 * (1 + xi[0])  # ∂N1/∂η
        # Knoten 2 (NE)
        gamma[q, 2, 0] = 0.25 * (1 + xi[1])  # ∂N2/∂ξ
        gamma[q, 2, 1] = 0.25 * (1 + xi[0])  # ∂N2/∂η
        # Knoten 3 (NW)
        gamma[q, 3, 0] = -0.25 * (1 + xi[1])  # ∂N3/∂ξ
        gamma[q, 3, 1] = 0.25 * (1 - xi[0])  # ∂N3/∂η

    return N, gamma
