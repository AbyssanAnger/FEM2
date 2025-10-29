import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_geometrie():
    a = a


def plot_results(solver, disp_scaling=1000):
    """
    Erstellt Plots für FEM-Ergebnisse: Undeformiert, Deformiert, Spannungen.

    Args:
        solver: Instanz von FEMSolver (benötigt self.x, self.u, self.elems, etc.)
        disp_scaling: Skalierung für Deformation (default: 1000)

    Returns:
        None (zeigt Plots an)
    """
    u_reshaped = solver.u.reshape(-1, solver.ndf)
    x_disp = solver.x + disp_scaling * u_reshaped

    # Voigt-Indizes für 2D-Spannungen (σxx, σyy, σxy)
    voigt = torch.tensor([[0, 0], [1, 1], [0, 1]])
    ei = torch.eye(3)

    # 2x2-Subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    indices = torch.tensor([0, 1, 2, 3, 0])  # Für geschlossene Elemente

    # 1. Undeformiertes Mesh
    ax = axs[0, 0]
    for e in range(solver.nel):
        els = solver.elems[e][indices]
        ax.plot(solver.x[els, 0], solver.x[els, 1], "k-")
    ax.plot(solver.x[:, 0], solver.x[:, 1], "ko", markersize=5)
    ax.set_title("Undeformiertes Mesh")
    ax.axis("equal")

    # 2. Deformiertes Mesh
    ax = axs[0, 1]
    for e in range(solver.nel):
        els = solver.elems[e][indices]
        ax.plot(x_disp[els, 0], x_disp[els, 1], "b-")
    ax.plot(x_disp[:, 0], x_disp[:, 1], "bo", markersize=5)
    ax.set_title("Deformiertes Mesh")
    ax.axis("equal")

    # 3. & 4. Spannungen an Gauss-Punkten (z. B. σxx und σxy)
    plotdata = torch.zeros(solver.nel * solver.nqp, 3)  # x, y, value
    components = ["xx", "xy"]  # Für die zwei Plots
    for i, comp_idx in enumerate([0, 2]):  # Index in voigt
        ax = axs[1, i]
        qp_idx = 0
        for e in range(solver.nel):
            xe = solver.x[solver.elems[e]].t()  # (ndm, nen)
            ue = u_reshaped[solver.elems[e]]  # (nen, ndf)
            for q in range(solver.nqp):
                # Gauss-Punkt-Position (undeformiert)
                xgauss = (xe @ solver.N[q]).squeeze()

                # Dehnung und Spannung berechnen
                gamma = solver.gamma[q]
                Je = xe @ gamma.t()
                detJe = torch.det(Je)
                if detJe <= 0:
                    print(f"Warning: Negative detJe in Element {e}, QP {q}")
                    continue
                invJe = torch.inverse(Je)
                G = gamma @ invJe
                h = torch.zeros(3, 3)
                h[: solver.ndm, : solver.ndm] = ue @ G
                eps = 0.5 * (h + h.t())
                stre = solver.material.stress_from_strain(
                    eps[: solver.ndm, : solver.ndm], ei
                )
                stre_val = stre[voigt[comp_idx, 0], voigt[comp_idx, 1]]

                plotdata[qp_idx] = torch.cat([xgauss, stre_val.unsqueeze(0)])
                qp_idx += 1

        # Scatter-Plot mit Farbkodierung
        max_val = torch.max(torch.abs(plotdata[:, 2])) + 1e-12
        scatter = ax.scatter(
            plotdata[:, 0],
            plotdata[:, 1],
            c=plotdata[:, 2] / max_val,
            s=100,
            cmap="jet",
        )
        ax.set_title(f"Spannung σ_{components[i]}")
        ax.axis("equal")
        plt.colorbar(scatter, ax=ax, label="Normalisiert")

    plt.tight_layout()
    plt.show()


def plot_boundary_conditions(mesh, ax=None):
    """
    Plottet Randbedingungen (Dirichlet: grün, Neumann: rot) auf ein Axes-Objekt.

    Args:
        mesh: Instanz von Mesh (benötigt self.x, self.drlt_mask, etc.)
        ax: Matplotlib-Axes (default: None → erstellt neues)

    Returns:
        ax: Das Axes-Objekt
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Fixierte DOFs (grüne Pfeile)
    drlt_nodes = torch.nonzero(mesh.drlt_mask.squeeze()).squeeze()
    for dof in drlt_nodes:
        node_id = dof // mesh.ndf  # Annahme ndf=2
        dir = dof % 2  # 0=x, 1=y
        x_pos = mesh.x[node_id]
        if dir == 0:  # x-Richtung
            ax.plot(x_pos[0] - 0.02, x_pos[1], "g>", markersize=10)
        else:  # y-Richtung
            ax.plot(x_pos[0], x_pos[1] - 0.02, "g^", markersize=10)

    # Kräfte (rote Pfeile)
    neum_nodes = torch.nonzero(mesh.neum_vals.squeeze() != 0).squeeze()
    for dof in neum_nodes:
        node_id = dof // 2
        dir = dof % 2
        x_pos = mesh.x[node_id]
        if dir == 0:
            ax.plot(x_pos[0] + 0.02, x_pos[1], "r>", markersize=10)
        else:
            ax.plot(x_pos[0], x_pos[1] + 0.02, "r^", markersize=10)

    ax.plot(mesh.x[:, 0], mesh.x[:, 1], "ko", markersize=5)  # Knoten
    ax.set_title("Randbedingungen")
    ax.axis("equal")
    return ax


# Weitere Hilfsfunktionen (erweitere bei Bedarf)
def timer(func):
    """Decorator für Timing von Funktionen."""
    import time as timemodule

    def wrapper(*args, **kwargs):
        start = timemodule.perf_counter()
        result = func(*args, **kwargs)
        end = timemodule.perf_counter()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result

    return wrapper


def save_results_to_csv(solver, filename="fem_results.csv"):
    """Speichert Verschiebungen u in eine CSV."""
    u_reshaped = solver.u.reshape(-1, solver.ndf).numpy()
    np.savetxt(filename, u_reshaped, delimiter=",", header="ux,uy", comments="")
    print(f"Results saved to {filename}")
