# solver.py
import torch
import math
from config import TDM, TORCH_THREADS
from src.material import IsotropicMaterial
from src.mesh import Mesh
from src.elements import gauss_quadrature, master_element  # Neu: Ausgelagert

torch.set_num_threads(TORCH_THREADS)


class FEMSolver:
    """Haupt-Solver: Integriert Mesh, Material und Analyse."""

    def __init__(
        self, mesh: Mesh, material: IsotropicMaterial, ndf=2, ndm=2, nen=4, nqp=4
    ):
        self.mesh = mesh
        self.material = material
        self.ndf = ndf
        self.ndm = ndm
        self.nen = nen
        self.tdm = TDM
        self.nqp = nqp
        self.qpt, self.w8 = gauss_quadrature(nqp=self.nqp, ndm=self.ndm)
        self.N, self.gamma = master_element(self.qpt, nen=self.nen, ndm=self.ndm)
        self.indices = torch.tensor([0, 1, 2, 3, 0])  # Für geschlossene Elemente

        # Initialisierung für Analyse
        self.nnp = self.mesh.nnp
        self.nel = self.mesh.nel
        self.u = None
        self.K = None
        self.fint = None
        self.fsur = None
        self.gdof = None
        self.voigt = torch.tensor([[0, 0], [1, 1], [0, 1]])  # Für 2D Voigt

    def assemble(self):
        """Montiert globale Steifigkeitsmatrix K und Kräftevektor F."""
        self.mesh.setup_bcs()
        self.u = torch.zeros(self.ndf * self.nnp, 1)

        # DOF-Mapping
        edof = torch.zeros(self.nel, self.ndf, self.nen, dtype=torch.long)
        gdof = torch.zeros(self.nel, self.ndf * self.nen, dtype=torch.long)
        for el in range(self.nel):
            for ien in range(self.nen):
                for idf in range(self.ndf):
                    edof[el, idf, ien] = self.ndf * self.mesh.elems[el, ien] + idf
            gdof[el] = edof[el].t().reshape(self.ndf * self.nen)

        self.gdof = gdof

        # Initialisieren
        Ke = torch.zeros(self.nen * self.ndf, self.nen * self.ndf)
        K = torch.zeros(self.nnp * self.ndf, self.nnp * self.ndf)
        finte = torch.zeros(self.nen * self.ndf, 1)
        fint = torch.zeros(self.ndf * self.nnp, 1)

        h = torch.zeros(self.tdm, self.tdm)

        for el in range(self.nel):
            xe = self.mesh.x[self.mesh.elems[el]].t()  # (ndm, nen)
            local_dofs = edof[el].flatten()
            ue_vec = self.u[local_dofs]
            ue = ue_vec.view(self.ndm, self.nen)

            Ke.zero_()
            finte.zero_()
            for q in range(self.nqp):
                N = self.N[q]
                gamma = self.gamma[q]

                Je = xe @ gamma
                detJe = torch.det(Je)
                if detJe <= 0:
                    raise ValueError(f"detJe <= 0 in Element {el}, QP {q}")

                dv = detJe * self.w8[q]
                invJe = torch.inverse(Je)
                G = gamma @ invJe

                h[: self.ndm, : self.ndm] = ue @ G
                eps = 0.5 * (h + h.t())[: self.ndm, : self.ndm]
                stre_2d = self.material.stress_from_strain(eps)

                # Interne Kräfte
                for A in range(self.nen):
                    G_A = G[A]
                    fintA = dv * (G_A @ stre_2d)
                    finte[A * self.ndf : (A + 1) * self.ndf] += fintA.unsqueeze(1)

                # Steifigkeit
                for A in range(self.nen):
                    for B in range(self.nen):
                        tmp = torch.tensordot(self.material.C4, G[B], dims=([3], [0]))
                        KAB = torch.tensordot(G[A], tmp, dims=([0], [0]))
                        Ke[
                            A * self.ndf : (A + 1) * self.ndf,
                            B * self.ndf : (B + 1) * self.ndf,
                        ] += (
                            dv * KAB
                        )

            # Globale Montage
            for i in range(self.gdof[el].size(0)):
                K[self.gdof[el][i], self.gdof[el]] += Ke[i]
            fint[self.gdof[el]] += finte

        self.K = K
        self.fint = fint
        self.fsur = self.mesh.neum_vals

    def solve(self):
        """Löst das lineare System Ku = F mit Penalty-Methode."""
        rsd = self.mesh.free_mask * (self.fsur - self.fint)
        K_tilde = self.K + self.mesh.drlt_matrix
        du = torch.linalg.solve(K_tilde, rsd)
        self.u += du
        self.u = (
            self.mesh.free_mask * self.u + self.mesh.drlt_mask * self.mesh.drlt_vals
        )
        print("u:", self.u)

        self.fext = self.K @ self.u
        self.frea = self.fext - self.fsur


def compute_modes(
    self, approach: str = "M_inv_K", num_modes: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Berechnet Eigenfrequenzen ω (rad/s) und -vektoren für freie Schwingungen.
    approach: 'K_inv_M' (1/√λ), 'M_inv_K' (√λ) oder 'lumped' (diagonale M).
    """
    if self.M is None:
        raise ValueError("Assembliere zuerst mit assemble() – M fehlt.")
    free_dofs = torch.nonzero(self.mesh.free_mask)[:, 0]
    K_free = self.K[free_dofs][:, free_dofs]
    M_free = self.M[free_dofs][:, free_dofs]
    if torch.det(M_free) <= 0:
        raise ValueError("M_free nicht positiv definit – prüfe Mesh/Material.")

    if approach == "K_inv_M":
        K_inv = torch.linalg.inv(K_free)
        A = K_inv @ M_free
        L, V = torch.linalg.eig(A)
        omega = torch.sort(1 / torch.sqrt(L.real))[0][:num_modes]
    elif approach == "M_inv_K":
        M_inv = torch.linalg.inv(M_free)
        A = M_inv @ K_free
        L, V = torch.linalg.eig(A)
        omega = torch.sort(torch.sqrt(L.real))[0][:num_modes]
    elif approach == "lumped":
        sum_m = torch.sum(M_free, dim=0)
        M_lump_inv = torch.diag(1 / sum_m)
        A = M_lump_inv @ K_free
        L, V = torch.linalg.eig(A)
        omega = torch.sort(torch.sqrt(L.real))[0][:num_modes]
    else:
        raise ValueError("approach muss 'K_inv_M', 'M_inv_K' oder 'lumped' sein.")

    print(f"Eigenfrequenzen ({approach}, erste {num_modes}): {omega}")
    return omega, V[:, :num_modes]  # ω sortiert, Vektoren (nicht normalisiert)


# NEU: Transient-Analyse
def solve_transient(
    self,
    dt: float,
    total_time: float,
    beta: float = 0.25,
    gamma: float = 0.5,
    f_ext_func=None,
) -> dict:
    """
    Newmark-Zeitintegration für dynamische Analyse.
    f_ext_func: Optional callable(t) -> f_ext (Tensor [total_dofs, 1]).
    Returns: {'t': list, 'u': list[Tensors], 'v': list[Tensors]}.
    """
    if self.M is None:
        raise ValueError("Assembliere zuerst mit assemble() M fehlt.")
    steps = int(total_time / dt)
    K_tilde = self.K + self.mesh.drlt_matrix
    u = (
        self.u.clone() if self.u is not None else torch.zeros(self.ndf * self.nnp, 1)
    )  # Von static starten
    v = torch.zeros_like(u)
    # Initial a_0 = M^{-1} (f_ext(0) - K_tilde u)
    f_ext0 = f_ext_func(0) if f_ext_func else torch.zeros_like(u)
    a = torch.linalg.solve(self.M, f_ext0 - K_tilde @ u)

    K_eff = K_tilde + (1 / (beta * dt**2)) * self.M
    history = {"t": [], "u": [], "v": []}

    for s in range(steps):
        t = s * dt
        f_ext = f_ext_func(t) if f_ext_func else torch.zeros_like(u)

        # Newmark-Update
        F1 = f_ext + self.M @ (
            (1 / (beta * dt**2)) * u
            + (1 / (beta * dt)) * v
            + ((1 - 2 * beta) / (2 * beta)) * a
        )
        u_new = torch.linalg.solve(K_eff, F1)
        a_new = (
            (1 / (beta * dt**2)) * (u_new - u)
            - (1 / (beta * dt)) * v
            - ((1 - 2 * beta) / (2 * beta)) * a
        )
        v_new = v + dt * ((1 - gamma) * a + gamma * a_new)

        u, v, a = u_new, v_new, a_new
        history["t"].append(t + dt)
        history["u"].append(u.clone())
        history["v"].append(v.clone())

        # Optional: Plot alle 10 Steps (importiere aus utils.py)
        # if s % 10 == 0:
        #     self._plot_step(u, s)  # Implementiere als private Methode

    print(
        f"Transient-Analyse abgeschlossen: {steps} Steps, finale ||u|| = {torch.norm(u):.2e}"
    )
    return history


def run(self):
    """Führt Assembly und Solve aus."""
    self.assemble()
    self.solve()
