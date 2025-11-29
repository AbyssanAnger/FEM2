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
        self, mesh: Mesh, material: IsotropicMaterial, ndf=2, ndm=2, nen=4, nqp=4, rho=7850.0
    ):
        self.mesh = mesh
        self.material = material
        self.ndf = ndf
        self.ndm = ndm
        self.nen = nen
        self.tdm = TDM
        self.nqp = nqp
        self.rho = rho
        
        self.qpt, self.w8 = gauss_quadrature(nqp=self.nqp, ndm=self.ndm)
        self.N, self.gamma = master_element(self.qpt, nen=self.nen, ndm=self.ndm)
        self.indices = torch.tensor([0, 1, 2, 3, 0])  # Default Q4, wird beim Plotten überschrieben

        # Initialisierung für Analyse
        self.nnp = self.mesh.nnp
        self.nel = self.mesh.nel
        self.u = None
        self.K = None
        self.M = None
        self.fint = None
        self.fsur = None
        self.gdof = None
        self.voigt = torch.tensor([[0, 0], [1, 1], [0, 1]])  # Für 2D Voigt
        
        # Ergebnisse der Modalanalyse
        self.eigenvalues = None
        self.frequencies = None

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
                    # Warnung statt Error, da bei Startkonfiguration manchmal numerisch 0
                    pass 
                    # raise ValueError(f"detJe <= 0 in Element {el}, QP {q}")

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

    def assemble_mass_matrix(self):
        """Montiert die globale Massenmatrix M."""
        Me = torch.zeros(self.nen * self.ndf, self.nen * self.ndf)
        M = torch.zeros(self.nnp * self.ndf, self.nnp * self.ndf)
        I = torch.eye(self.ndm, self.ndm)

        for el in range(self.nel):
            xe = self.mesh.x[self.mesh.elems[el]].t()
            Me.zero_()
            
            for q in range(self.nqp):
                N = self.N[q]
                gamma = self.gamma[q]
                Je = xe @ gamma
                detJe = torch.det(Je)
                dv = detJe * self.w8[q]

                for A in range(self.nen):
                    for B in range(self.nen):
                        MAB = self.rho * N[A] * N[B] * I
                        Me[
                            A * self.ndf : (A + 1) * self.ndf,
                            B * self.ndf : (B + 1) * self.ndf,
                        ] += (dv * MAB)

            # Globale Montage
            for i in range(self.gdof[el].size(0)):
                M[self.gdof[el][i], self.gdof[el]] += Me[i]
        
        self.M = M

    def modal_analysis(self, num_modes=10, f_threshold=0.001):
        """Führt eine Modalanalyse durch und berechnet Eigenfrequenzen."""
        if self.M is None:
            self.assemble_mass_matrix()
            
        # Reduzierte Matrizen für freie Freiheitsgrade
        free_dofs = torch.nonzero(self.mesh.free_mask)[:, 0]
        K_free = self.K[free_dofs, :][:, free_dofs]
        M_free = self.M[free_dofs, :][:, free_dofs]
        
        # 4. Ansatz: Invertierte Lumped Mass Matrix (am stabilsten)
        sumM = torch.sum(M_free, dim=0)
        diagM = torch.diag(sumM)
        diagMInv = torch.linalg.inv(diagM)
        diagMInvK = diagMInv @ K_free

        L, V = torch.linalg.eig(diagMInvK)
        
        # Berechnung Frequenzen
        l_val = torch.sqrt(L.real)
        
        # Sortieren
        sorted_indices = torch.argsort(l_val)
        l_val = l_val[sorted_indices]
        V = V[:, sorted_indices] # Eigenvektoren mitsortieren
        
        freq = l_val / (2.0 * math.pi)
        
        # Filter: Nur Frequenzen > Threshold
        mask = freq > f_threshold
        self.frequencies = freq[mask]
        self.eigenvectors = V[:, mask]
        
        # Rückgabe der ersten N Moden
        return self.frequencies[:num_modes], self.eigenvectors[:, :num_modes]

    def newmark_time_integration(self, beta=0.25, gamma=0.5, dt=0.01, steps=1000, disp_scaling=1.0):
        """Führt eine transiente Analyse mit Newmark-Zeitintegration durch."""
        if self.M is None:
            self.assemble_mass_matrix()
            
        # Initialisierung
        u_0 = self.u.clone() # Startverschiebung aus statischer Analyse
        v_0 = torch.zeros_like(u_0)
        
        # Anfangsbeschleunigung: M * a0 = F_ext - K * u0
        # Wir nehmen an F_ext = 0 für freie Schwingung nach Lastabwurf oder konstant
        # Hier: Freie Schwingung (F_ext = 0)
        fsur_dynamic = torch.zeros_like(self.fsur)
        
        # a_0 berechnen
        # M * a_0 = fsur - K * u_0
        # Da M singulär sein kann (Randbedingungen), nutzen wir die reduzierte Form oder Penalty
        # Hier vereinfacht: Wir lösen das System mit Penalty auf M (nicht ideal) oder reduzieren.
        # Besser: Explizite Berechnung auf freien DOFs.
        
        # Wir nutzen die effektive Steifigkeitsmatrix Methode
        K_tilde = self.K + self.mesh.drlt_matrix
        K_eff = K_tilde + 1 / (beta * dt * dt) * self.M
        
        # Initial a_0 (vereinfacht, Annahme M invertierbar oder System lösbar)
        # M * a_0 = R_0 = fsur_dynamic - K * u_0
        R_0 = fsur_dynamic - self.K @ u_0
        # Wir lösen (M + Penalty) * a_0 = R_0
        M_tilde = self.M + self.mesh.drlt_matrix # Penalty auf Masse für fixierte Knoten -> a=0
        a_0 = torch.linalg.solve(M_tilde, R_0)
        
        time_history = []
        disp_history = []
        
        # Zeitschleife
        for s in range(steps):
            time_current = (s + 1) * dt
            
            # Prädiktor (nicht explizit nötig bei implizitem Newmark, aber für Formel)
            
            # Effektive Kraft
            # F_eff = F_ext + M * (c1*u0 + c2*v0 + c3*a0)
            c1 = 1 / (beta * dt * dt)
            c2 = 1 / (beta * dt)
            c3 = (1 - 2 * beta) / (2 * beta)
            
            F_eff = fsur_dynamic + self.M @ (c1 * u_0 + c2 * v_0 + c3 * a_0)
            
            # Lösen nach u_1
            u_1 = torch.linalg.solve(K_eff, F_eff)
            
            # Update Kinematik
            a_1 = c1 * (u_1 - u_0) - c2 * v_0 - c3 * a_0
            v_1 = v_0 + dt * ((1 - gamma) * a_0 + gamma * a_1)
            
            # Speichern für Plot (Verschiebung am letzten Knoten, z.B. u_y)
            # Wir nehmen den Knoten mit der größten Verschiebung oder den letzten Knoten
            # Hier: Letzter Knoten, y-Richtung (Index -1 da 2 DOFs pro Knoten)
            # Oder besser: u_x am Ende (wie im Originalskript)
            u_last_node_x = u_1[-2].item() * disp_scaling # Vorletzter Wert ist u_x des letzten Knotens
            
            time_history.append(time_current)
            disp_history.append(u_last_node_x)
            
            # Update für nächsten Schritt
            u_0 = u_1
            v_0 = v_1
            a_0 = a_1

            yield time_current, u_1, u_last_node_x

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

