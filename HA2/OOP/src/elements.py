import torch
import math


def gauss_quadrature(nqp=4, ndm=2):
    """Definiert Gauss-Punkte und -Gewichte."""
    if nqp == 9:
        # 3x3 Gauss-Quadratur für Q8
        qpt = torch.zeros(nqp, ndm)
        w8 = torch.zeros(nqp, 1)
        a = math.sqrt(3.0 / 5.0)
        w1 = 5.0 / 9.0
        w2 = 8.0 / 9.0

        # Definition der 9 Gauß-Punkte
        qpt[0, :] = torch.tensor([-a, -a])
        qpt[1, :] = torch.tensor([0, -a])
        qpt[2, :] = torch.tensor([a, -a])
        qpt[3, :] = torch.tensor([-a, 0])
        qpt[4, :] = torch.tensor([0, 0])
        qpt[5, :] = torch.tensor([a, 0])
        qpt[6, :] = torch.tensor([-a, a])
        qpt[7, :] = torch.tensor([0, a])
        qpt[8, :] = torch.tensor([a, a])

        # Gewichte
        w8[0] = w1 * w1
        w8[1] = w1 * w2
        w8[2] = w1 * w1
        w8[3] = w2 * w1
        w8[4] = w2 * w2
        w8[5] = w2 * w1
        w8[6] = w1 * w1
        w8[7] = w1 * w2
        w8[8] = w1 * w1
        
        return qpt, w8
    else:
        # Standard 2x2 Gauss-Quadratur für Q4
        a = math.sqrt(3) / 3
        qpt = torch.tensor([[-a, -a], [a, -a], [-a, a], [a, a]])
        w8 = torch.ones(nqp, 1)
        return qpt, w8


def master_element(qpt, nen=4, ndm=2):
    """Berechnet Shape-Funktionen N und Ableitungen gamma im Master-Element."""
    nqp = qpt.shape[0]
    N = torch.zeros(nqp, nen)
    gamma = torch.zeros(nqp, nen, ndm)
    
    for q in range(nqp):
        xi = qpt[q]
        e, n = xi[0], xi[1]
        
        if nen == 8:
            # Q8 Shape Functions (Serendipity)
            # Eckknoten
            N[q, 0] = 0.25 * (1 - e) * (1 - n) * (-e - n - 1)
            N[q, 1] = 0.25 * (1 + e) * (1 - n) * (e - n - 1)
            N[q, 2] = 0.25 * (1 + e) * (1 + n) * (e + n - 1)
            N[q, 3] = 0.25 * (1 - e) * (1 + n) * (-e + n - 1)
            # Mittenknoten
            N[q, 4] = 0.5 * (1 - e * e) * (1 - n)
            N[q, 5] = 0.5 * (1 + e) * (1 - n * n)
            N[q, 6] = 0.5 * (1 - e * e) * (1 + n)
            N[q, 7] = 0.5 * (1 - e) * (1 - n * n)

            # Ableitungen (d/de, d/dn)
            # d/de
            gamma[q, 0, 0] = 0.25 * (1 - n) * (-1) * (-e - n - 1) + 0.25 * (1 - e) * (1 - n) * (-1)
            gamma[q, 1, 0] = 0.25 * (1 - n) * (1) * (e - n - 1) + 0.25 * (1 + e) * (1 - n) * (1)
            gamma[q, 2, 0] = 0.25 * (1 + n) * (1) * (e + n - 1) + 0.25 * (1 + e) * (1 + n) * (1)
            gamma[q, 3, 0] = 0.25 * (1 + n) * (-1) * (-e + n - 1) + 0.25 * (1 - e) * (1 + n) * (-1)
            gamma[q, 4, 0] = 0.5 * (-2 * e) * (1 - n)
            gamma[q, 5, 0] = 0.5 * (1) * (1 - n * n)
            gamma[q, 6, 0] = 0.5 * (-2 * e) * (1 + n)
            gamma[q, 7, 0] = 0.5 * (-1) * (1 - n * n)
            # d/dn
            gamma[q, 0, 1] = 0.25 * (1 - e) * (-1) * (-e - n - 1) + 0.25 * (1 - e) * (1 - n) * (-1)
            gamma[q, 1, 1] = 0.25 * (1 + e) * (-1) * (e - n - 1) + 0.25 * (1 + e) * (1 - n) * (-1)
            gamma[q, 2, 1] = 0.25 * (1 + e) * (1) * (e + n - 1) + 0.25 * (1 + e) * (1 + n) * (1)
            gamma[q, 3, 1] = 0.25 * (1 - e) * (1) * (-e + n - 1) + 0.25 * (1 - e) * (1 + n) * (1)
            gamma[q, 4, 1] = 0.5 * (1 - e * e) * (-1)
            gamma[q, 5, 1] = 0.5 * (1 + e) * (-2 * n)
            gamma[q, 6, 1] = 0.5 * (1 - e * e) * (1)
            gamma[q, 7, 1] = 0.5 * (1 - e) * (-2 * n)
            
        else:
            # Q4 Shape Functions (Bilinear)
            # Shape Functions
            N[q, 0] = 0.25 * (1 - e) * (1 - n)  # SW
            N[q, 1] = 0.25 * (1 + e) * (1 - n)  # SE
            N[q, 2] = 0.25 * (1 + e) * (1 + n)  # NE
            N[q, 3] = 0.25 * (1 - e) * (1 + n)  # NW

            # Ableitungen (∂N/∂ξ, ∂N/∂η)
            gamma[q, 0, 0] = -0.25 * (1 - n)  # ∂N0/∂ξ
            gamma[q, 0, 1] = -0.25 * (1 - e)  # ∂N0/∂η
            gamma[q, 1, 0] = 0.25 * (1 - n)   # ∂N1/∂ξ
            gamma[q, 1, 1] = -0.25 * (1 + e)  # ∂N1/∂η
            gamma[q, 2, 0] = 0.25 * (1 + n)   # ∂N2/∂ξ
            gamma[q, 2, 1] = 0.25 * (1 + e)   # ∂N2/∂η
            gamma[q, 3, 0] = -0.25 * (1 + n)  # ∂N3/∂ξ
            gamma[q, 3, 1] = 0.25 * (1 - e)   # ∂N3/∂η

    return N, gamma


def get_plot_indices(nen):
    """Gibt die Indizes für das Plotten der Elementkanten zurück."""
    if nen == 8:
        # Q8: 4 Ecken + 4 Mittenknoten -> Plotten der Kanten
        # Reihenfolge im Element: 0-1-2-3 (Ecken), 4-5-6-7 (Mitten)
        # Plot-Reihenfolge: 0 -> 4 -> 1 -> 5 -> 2 -> 6 -> 3 -> 7 -> 0
        return torch.tensor([0, 4, 1, 5, 2, 6, 3, 7, 0])
    else:
        # Q4: 0 -> 1 -> 2 -> 3 -> 0
        return torch.tensor([0, 1, 2, 3, 0])
