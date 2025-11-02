import matplotlib.pyplot as plt
import torch
import numpy as np

# if TOPLOT:
#     plot_beam(self)
#     plot_results(self)


def plot_beam(solver, figsize=(10, 4)):
    """
    Plottet das FEM-Mesh eines Balkens.

    Parameter:
    - coords: Knoten-Koordinaten (aus generate_beam)
    - elems: Element-Konnektivitäten (aus generate_beam)
    - drlt: Dirichlet-BCs (aus generate_beam)
    - neum: Neumann-BCs (aus generate_beam)
    - l, b: Optional für Labels/Titel (falls nicht übergeben, aus coords abgeleitet)
    - nx, ny: Optional für Titel (Anzahl Elemente)
    - figsize: Größe der Figur (Standard: breiter für lange Balken)

    Returns: None (zeigt Plot an)
    """
    if l is None or b is None:
        l = coords[:, 0].max() - coords[:, 0].min()
        b = coords[:, 1].max() - coords[:, 1].min()
    if nx is None or ny is None:
        nx = (
            len(elems) // (np.max(elems[:, 1]) - np.min(elems[:, 0]) + 1)
            if elems.size > 0
            else 2
        )  # Grobe Schätzung
        ny = len(elems) // nx if elems.size > 0 else 2

    fig, ax = plt.subplots(figsize=figsize)

    # Aspekt-Ratio proportional zum Balken (nicht 'equal' – für schmale Balken essenziell!)
    aspect = b / l
    ax.set_aspect(aspect)

    # Knoten plotten (kleine blaue Punkte)
    ax.scatter(coords[:, 0], coords[:, 1], c="blue", s=5, label="Knoten")

    # Elemente plotten (blaue Linien)
    for elem in elems:
        pts = coords[elem]
        ax.plot(pts[:, 0], pts[:, 1], "b-", linewidth=1)

    # Dirichlet-BCs markieren (rote Dreiecke)
    drlt_nodes = drlt[:, 0].astype(int)
    seen_drlt = False
    for node_id in drlt_nodes:
        if not seen_drlt:
            ax.scatter(
                coords[node_id, 0],
                coords[node_id, 1],
                c="red",
                marker="^",
                s=50,
                label="Dirichlet-BC",
            )
            seen_drlt = True
        else:
            ax.scatter(
                coords[node_id, 0], coords[node_id, 1], c="red", marker="^", s=50
            )

    # Neumann-BCs markieren (grüne Dreiecke) + Last-Werte als Labels
    neum_nodes = neum[:, 0].astype(int)
    seen_neum = False
    for i, node_id in enumerate(neum_nodes):
        load_value = neum[i, 2]
        if not seen_neum:
            ax.scatter(
                coords[node_id, 0],
                coords[node_id, 1],
                c="green",
                marker="v",
                s=50,
                label="Neumann-BC",
            )
            seen_neum = True
        else:
            ax.scatter(
                coords[node_id, 0], coords[node_id, 1], c="green", marker="v", s=50
            )
        # Kleines Label für den Last-Wert (wissenschaftliche Notation für große Zahlen)
        ax.annotate(
            f"{load_value:.1e}",
            (coords[node_id, 0], coords[node_id, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6,
            color="green",
        )

    # Achsen und Labels
    ax.set_xlabel("x (Länge)")
    ax.set_ylabel("y (Breite)")
    ax.set_title(
        f"Finite-Elemente-Mesh des Balkens (l={l:.3f}, b={b:.3f}, nx={nx}, ny={ny})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()  # Passt Ränder an
    plt.show()


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
    x_disp = solver.coords + disp_scaling * u_reshaped

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


# def plot_boundary_conditions(mesh, ax=None):
#     """
#     Plottet Randbedingungen (Dirichlet: grün, Neumann: rot) auf ein Axes-Objekt.

#     Args:
#         mesh: Instanz von Mesh (benötigt self.x, self.drlt_mask, etc.)
#         ax: Matplotlib-Axes (default: None → erstellt neues)

#     Returns:
#         ax: Das Axes-Objekt
#     """
#     if ax is None:
#         fig, ax = plt.subplots()

#     # Fixierte DOFs (grüne Pfeile)
#     drlt_nodes = torch.nonzero(mesh.drlt_mask.squeeze()).squeeze()
#     for dof in drlt_nodes:
#         node_id = dof // mesh.ndf  # Annahme ndf=2
#         dir = dof % 2  # 0=x, 1=y
#         x_pos = mesh.x[node_id]
#         if dir == 0:  # x-Richtung
#             ax.plot(x_pos[0] - 0.02, x_pos[1], "g>", markersize=10)
#         else:  # y-Richtung
#             ax.plot(x_pos[0], x_pos[1] - 0.02, "g^", markersize=10)

#     # Kräfte (rote Pfeile)
#     neum_nodes = torch.nonzero(mesh.neum_vals.squeeze() != 0).squeeze()
#     for dof in neum_nodes:
#         node_id = dof // 2
#         dir = dof % 2
#         x_pos = mesh.x[node_id]
#         if dir == 0:
#             ax.plot(x_pos[0] + 0.02, x_pos[1], "r>", markersize=10)
#         else:
#             ax.plot(x_pos[0], x_pos[1] + 0.02, "r^", markersize=10)

#     ax.plot(mesh.x[:, 0], mesh.x[:, 1], "ko", markersize=5)  # Knoten
#     ax.set_title("Randbedingungen")
#     ax.axis("equal")
#     return ax


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
