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

    def run(self):
        """Führt Assembly und Solve aus."""
        self.assemble()
        self.solve()
