import torch
from config import TOPLOT  # Für Plot-Flag

from .mesh import Mesh
from .material import IsotropicMaterial
from .elements import gauss_quadrature, master_element
from .utils import plot_beam, plot_results


class FEMSolver:
    """Haupt-Solver: Integriert alles."""

    def __init__(self, mesh: Mesh, material: IsotropicMaterial):
        self.mesh = mesh
        self.material = material
        self.ndf = 2
        self.ndm = 2
        self.nen = 4
        self.tdm = 2
        self.qpt, self.w8 = gauss_quadrature()  # Bereits Float64
        self.N, self.gamma = master_element(self.qpt)  # Bereits Float64, Listen-frei
        self.nqp = self.qpt.shape[0]  # Neu: Für konsistente Loops

        # Initialisiere nach Setup (wird in run() gesetzt)
        self.nnp = self.mesh.nnp
        self.nel = self.mesh.nel
        self.u = None  # Wird in assemble() gesetzt
        self.K = None
        self.fint = None
        self.fsur = None
        self.gdof = None  # Globales DOF-Mapping

    def assemble(self):
        """Montiert globale Steifigkeitsmatrix K und Kräftevektor F."""
        dtype = torch.float64  # Explizit Double Precision (Float64) für alle Tensoren
        self.u = torch.zeros(self.ndf * self.nnp, 1, dtype=dtype)

        # DOF-Mapping
        edof = torch.zeros(self.nel, self.ndf, self.nen, dtype=torch.long)
        gdof = torch.zeros(self.nel, self.ndf * self.nen, dtype=torch.long)
        for el in range(self.nel):
            for ien in range(self.nen):
                for idf in range(self.ndf):
                    edof[el, idf, ien] = self.ndf * self.mesh.elems[el, ien] + idf
            # Transpose-Fix: .t() für (2,4) -> (4,2), dann reshape(8)
            gdof[el] = edof[el].t().reshape(self.ndf * self.nen)

        self.gdof = gdof  # Speichere für spätere Nutzung

        # Initialisieren
        Ke = torch.zeros(self.nen * self.ndf, self.nen * self.ndf, dtype=dtype)
        K = torch.zeros(self.nnp * self.ndf, self.nnp * self.ndf, dtype=dtype)
        finte = torch.zeros(self.nen * self.ndf, 1, dtype=dtype)
        fint = torch.zeros(self.ndf * self.nnp, 1, dtype=dtype)

        ei = torch.eye(self.ndm, dtype=dtype)
        h = torch.zeros(self.tdm, self.tdm, dtype=dtype)

        for el in range(self.nel):

            xe = (
                self.mesh.x[self.mesh.elems[el]].t().double()
            )  # (ndm, nen) = (2,4), explizit zu Float64
            local_dofs = edof[el].flatten()  # Neu: Flatten für sichere Indizierung (8,)
            ue_vec = self.u[local_dofs]  # (8,1), Float64
            ue = ue_vec.view(self.ndm, self.nen)  # Fix: (2,4) – DOFs als (ndm, nen)

            Ke.zero_()
            finte.zero_()
            for q in range(self.nqp):
                N = self.N[q]
                gamma = self.gamma[q]  # (4,2) – jetzt Float64

                # Fix: Kein .t() – (2,4) @ (4,2) = (2,2), beide Float64
                Je = xe @ gamma
                detJe = torch.det(Je)
                print(
                    f"Debug Element {el}, QP {q}: detJe = {detJe:.6f}, Je = {Je}"
                )  # Temporär für Debug
                if detJe <= 0:
                    raise ValueError(
                        f"Error: detJe = {detJe} <= 0 in Element {el}, QP {q}"
                    )

                dv = detJe * self.w8[q]
                invJe = torch.inverse(Je)
                G = gamma @ invJe  # (4,2) @ (2,2) = (4,2)

                # Fix: ue (2,4) @ G (4,2) = (2,2)
                h[: self.ndm, : self.ndm] = ue @ G
                eps = 0.5 * (h + h.t())[: self.ndm, : self.ndm]
                stre = self.material.stress_from_strain(eps, ei)

                # Interne Kräfte
                for A in range(self.nen):
                    G_A = G[A]  # (2,)
                    fintA = dv * (G_A @ stre)  # (1,2) @ (2,2) -> (2,)
                    finte[A * self.ndf : (A + 1) * self.ndf] += fintA.unsqueeze(1)

                # Steifigkeit
                for A in range(self.nen):
                    for B in range(self.nen):
                        # Tensordot für G_A : C : G_B -> (2,2)
                        tmp = torch.tensordot(
                            self.material.C4, G[B], dims=([3], [0])
                        )  # C4 (2,2,2,2) @ G_B (2,) -> (2,2,2)
                        KAB = torch.tensordot(
                            G[A], tmp, dims=([0], [0])
                        )  # G_A (2,) @ (2,2,2) -> (2,2)
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
        """Löst das lineare System Ku = F mit Straf-Methode für BCs."""
        rsd = self.mesh.free_mask * (self.fsur - self.fint)
        K_tilde = self.K + self.mesh.drlt_matrix
        du = torch.linalg.solve(K_tilde, rsd)
        self.u += du
        self.u = (
            self.mesh.free_mask * self.u + self.mesh.drlt_mask * self.mesh.drlt_vals
        )
        print("u:", self.u)

        # Reaktionskräfte (optional)
        self.fext = self.K @ self.u
        self.frea = self.fext - self.fsur

    def run(self, drlt, neum):
        self.mesh.setup_bcs(drlt, neum)
        self.assemble()
        self.solve()
