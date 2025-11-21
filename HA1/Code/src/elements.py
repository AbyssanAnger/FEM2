import torch
import math


def gauss_quadrature(nqp=4, ndm=2):
    """Definiert Gauss-Punkte und -Gewichte für 2x2-Integration."""
    a = math.sqrt(3) / 3
    qpt = torch.tensor([[-a, -a], [a, -a], [-a, a], [a, a]])
    w8 = torch.ones(nqp, 1)
    return qpt, w8


def master_element(qpt, nen=4, ndm=2):
    """Berechnet Shape-Funktionen N und Ableitungen gamma im Master-Element (bilinear Q4)."""
    nqp = qpt.shape[0]
    N = torch.zeros(nqp, nen)
    gamma = torch.zeros(nqp, nen, ndm)
    for q in range(nqp):
        xi = qpt[q]
        # Shape Functions
        N[q, 0] = 0.25 * (1 - xi[0]) * (1 - xi[1])  # SW
        N[q, 1] = 0.25 * (1 + xi[0]) * (1 - xi[1])  # SE
        N[q, 2] = 0.25 * (1 + xi[0]) * (1 + xi[1])  # NE
        N[q, 3] = 0.25 * (1 - xi[0]) * (1 + xi[1])  # NW

        # Ableitungen (∂N/∂ξ, ∂N/∂η)
        gamma[q, 0, 0] = -0.25 * (1 - xi[1])  # ∂N0/∂ξ
        gamma[q, 0, 1] = -0.25 * (1 - xi[0])  # ∂N0/∂η
        gamma[q, 1, 0] = 0.25 * (1 - xi[1])  # ∂N1/∂ξ
        gamma[q, 1, 1] = -0.25 * (1 + xi[0])  # ∂N1/∂η
        gamma[q, 2, 0] = 0.25 * (1 + xi[1])  # ∂N2/∂ξ
        gamma[q, 2, 1] = 0.25 * (1 + xi[0])  # ∂N2/∂η
        gamma[q, 3, 0] = -0.25 * (1 + xi[1])  # ∂N3/∂ξ
        gamma[q, 3, 1] = 0.25 * (1 - xi[0])  # ∂N3/∂η

    return N, gamma
