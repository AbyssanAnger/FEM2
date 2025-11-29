# src/plotting.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import time as timemodule
import math
from config import LENGTH, HEIGHT, WIDTH, FORCE, TOTAL_FORCE
from src.analytics import analytical_sigma_xx_midspan
from src.elements import get_plot_indices


def plot_mesh(ax, mesh, elems, x, indices):
    for e in range(mesh.nel):
        els = torch.index_select(elems[e, :], 0, indices)
        ax.plot(x[els, 0], x[els, 1], "k-", linewidth=0.5)
    ax.plot(x[:, 0], x[:, 1], "ko", markersize=2)
    ax.axis("equal")


def _compute_element_dofs(ndf, elem_nodes):
    local = []
    for ien in range(elem_nodes.numel()):
        node = int(elem_nodes[ien].item())
        for idf in range(ndf):
            local.append(node * ndf + idf)
    return torch.tensor(local, dtype=torch.long)


def animate_eigenmode(solver, frequencies, eigenvectors, mode_index=1, duration=5.0):
    """Animiert die N-te Eigenmode (1-based index)."""
    plt.ion()
    
    x = solver.mesh.x
    elems = solver.mesh.elems
    indices = get_plot_indices(solver.nen)
    
    # Rekonstruktion der vollen Vektoren
    free_dofs = torch.nonzero(solver.mesh.free_mask)[:, 0]
    
    # Check index
    idx = mode_index - 1
    if idx < 0 or idx >= len(frequencies):
        print(f"Mode {mode_index} nicht verfügbar.")
        return

    mode_shape = eigenvectors[:, idx].real
    full_mode = torch.zeros(solver.nnp * solver.ndf, 1)
    full_mode[free_dofs, 0] = mode_shape
    u_mode = full_mode.reshape(-1, solver.ndf)
    
    # Automatische Skalierung
    max_disp = torch.max(torch.abs(u_mode))
    if max_disp > 0:
        scale_factor = 0.2 * LENGTH / max_disp 
    else:
        scale_factor = 1.0
        
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    freq = frequencies[idx]
    ax.set_title(f"Eigenmode {mode_index} ({freq:.2f} Hz) - Animiert")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    
    # Statische Elemente (unverformt)
    for e in range(solver.mesh.nel):
        els = torch.index_select(elems[e, :], 0, indices)
        ax.plot(x[els, 0], x[els, 1], "k--", linewidth=0.2, alpha=0.5)
        
    # Dynamische Elemente initialisieren
    lines = []
    for e in range(solver.mesh.nel):
        els = torch.index_select(elems[e, :], 0, indices)
        line, = ax.plot([], [], 'b-', linewidth=0.5)
        lines.append(line)
        
    ax.set_xlim([-0.1 * LENGTH, 1.1 * LENGTH])
    ax.set_ylim([-5 * HEIGHT, 5 * HEIGHT])
    
    print(f"Starte Animation für Mode {mode_index}...")
    
    start_time = timemodule.perf_counter()
    while True:
        current_time = timemodule.perf_counter() - start_time
        if current_time > duration:
            break
            
        # Oszillation: sin(2*pi*f*t) - aber verlangsamt für Visualisierung
        # Wir nehmen eine feste visuelle Frequenz, z.B. 1 Hz, damit man es sieht
        factor = math.sin(2 * math.pi * 1.0 * current_time) 
        
        x_mode = x + u_mode * scale_factor * factor
        
        for i, line in enumerate(lines):
            els = torch.index_select(elems[i, :], 0, indices)
            line.set_data(x_mode[els, 0], x_mode[els, 1])
            
        plt.pause(0.01)
        
    plt.ioff()
    plt.show()


def animate_time_integration(solver, generator, disp_scaling=1.0):
    """Animiert die transiente Analyse."""
    plt.ion()  # Interaktiver Modus an
    
    fig = plt.figure(figsize=(12, 6))
    ax_mesh = plt.subplot(1, 2, 1)
    ax_hist = plt.subplot(1, 2, 2)
    
    # Setup Mesh Plot
    x = solver.mesh.x
    elems = solver.mesh.elems
    indices = get_plot_indices(solver.nen)
    
    # Statische Elemente plotten (unverformt als Referenz)
    # plot_mesh(ax_mesh, solver.mesh, elems, x, indices) # Optional: Unverformtes Netz im Hintergrund
    
    # Dynamische Elemente initialisieren
    lines = []
    for e in range(solver.mesh.nel):
        els = torch.index_select(elems[e, :], 0, indices)
        line, = ax_mesh.plot([], [], 'b-', linewidth=0.5)
        lines.append(line)
        
    ax_mesh.set_xlim([-0.1 * LENGTH, 1.1 * LENGTH])
    ax_mesh.set_ylim([-5 * HEIGHT, 5 * HEIGHT])
    ax_mesh.set_aspect('equal')
    ax_mesh.set_title("Verformung (animiert)")
    ax_mesh.grid(True)
    
    # Setup History Plot
    line_hist, = ax_hist.plot([], [], 'b-')
    ax_hist.set_title("Verschiebung u_x am Ende")
    ax_hist.set_xlabel("Zeit [s]")
    ax_hist.set_ylabel("Verschiebung [m]")
    ax_hist.grid(True)
    
    time_data = []
    disp_data = []
    
    print("Starte Animation...")
    
    try:
        for step, (t, u, u_val) in enumerate(generator):
            # Update History
            time_data.append(t)
            disp_data.append(u_val)
            
            line_hist.set_data(time_data, disp_data)
            ax_hist.relim()
            ax_hist.autoscale_view()
            
            # Update Mesh
            u_reshaped = u.view(-1, solver.ndf)
            x_disped = x + disp_scaling * u_reshaped
            
            for i, line in enumerate(lines):
                els = torch.index_select(elems[i, :], 0, indices)
                line.set_data(x_disped[els, 0], x_disped[els, 1])
            
            ax_mesh.set_title(f"Zeit: {t:.4f} s")
            
            plt.pause(0.001)
            
    except KeyboardInterrupt:
        print("Animation abgebrochen.")
        
    plt.ioff() # Interaktiver Modus aus
    plt.show() # Plot offen lassen


def plot_results(solver, disp_scaling=50):
    """Replicates extended post-processing from the original script.

    Plots:
    - Unverformtes Netz
    - Spannung XX, Spannung YY, Von-Mises-Spannung
    - Verschiebung u_x, Verschiebung u_y
    - Dehnung XX
    - Cross-section stress at x = L/2
    """
    x = solver.mesh.x
    elems = solver.mesh.elems
    nnp = solver.mesh.nnp
    nel = solver.mesh.nel
    ndf, ndm, nen, nqp = solver.ndf, solver.ndm, solver.nen, solver.nqp
    
    # Get correct plot indices for Q4 or Q8
    plot_indices = get_plot_indices(nen)

    u = solver.u.view(-1, ndf)
    x_disped = x + disp_scaling * u

    titles = [
        "Spannung XX",
        "Spannung YY",
        "Von-Mises-Spannung",
        "Verschiebung u_x",
        "Verschiebung u_y",
        "Dehnung XX",
    ]

    # For nodal extrapolation/averaging
    nodal_sums = torch.zeros(nnp, len(titles), dtype=x.dtype)
    nodal_counts = torch.zeros(nnp, len(titles), dtype=x.dtype)

    # Precompute N at gauss points (nqp x nen)
    N_qp = solver.N  # (nqp, nen)
    gamma_qp = solver.gamma  # (nqp, nen, ndm)
    N_at_gps = torch.vstack([N_qp[q] for q in range(nqp)])  # (nqp, nen)
    try:
        # Least-Squares Extrapolation: (N^T N)^-1 N^T
        extrap = torch.linalg.inv(N_at_gps.T @ N_at_gps) @ N_at_gps.T
    except RuntimeError:
        extrap = torch.linalg.pinv(N_at_gps)

    # Loop elements to compute gauss point fields and extrapolate to nodes
    for e in range(nel):
        elem_nodes = elems[e]
        local_dofs = _compute_element_dofs(ndf, elem_nodes)
        ue_vec = solver.u[local_dofs].view(nen, ndf).t()  # (ndm, nen)
        xe = x[elem_nodes].t()  # (ndm, nen)

        # Per-gauss arrays
        s_xx = torch.zeros(nqp, dtype=x.dtype)
        s_yy = torch.zeros(nqp, dtype=x.dtype)
        s_vm = torch.zeros(nqp, dtype=x.dtype)
        eps_xx = torch.zeros(nqp, dtype=x.dtype)

        for q in range(nqp):
            gamma = gamma_qp[q]  # (nen, ndm)
            Je = xe @ gamma
            detJe = torch.det(Je)
            if detJe <= 0:
                continue
            invJe = torch.inverse(Je)
            G = gamma @ invJe  # (nen, ndm)

            h = ue_vec @ G  # (ndm, ndm)
            eps = 0.5 * (h + h.t())[:ndm, :ndm]  # 2x2
            stre = solver.material.stress_from_strain(eps)  # 2x2

            s11 = stre[0, 0]
            s22 = stre[1, 1]
            s12 = stre[0, 1]
            s33 = torch.tensor(0.0, dtype=x.dtype)  # plane stress
            sigma_vm = torch.sqrt(
                0.5
                * (
                    (s11 - s22) ** 2
                    + (s22 - s33) ** 2
                    + (s33 - s11) ** 2
                    + 6.0 * (s12**2)
                )
            )

            s_xx[q] = s11
            s_yy[q] = s22
            s_vm[q] = sigma_vm
            eps_xx[q] = eps[0, 0]

        # Extrapolate gp -> nodes for stress/strain fields
        nodal_vals_elem = torch.stack(
            [
                extrap @ s_xx,
                extrap @ s_yy,
                extrap @ s_vm,
                extrap @ eps_xx,  # will map into Dehnung XX slot later
            ]
        )  # (4, nen)

        # Accumulate
        for i_local, node in enumerate(elem_nodes):
            node_idx = int(node.item())
            nodal_sums[node_idx, 0] += nodal_vals_elem[0, i_local]
            nodal_sums[node_idx, 1] += nodal_vals_elem[1, i_local]
            nodal_sums[node_idx, 2] += nodal_vals_elem[2, i_local]
            nodal_sums[node_idx, 5] += nodal_vals_elem[3, i_local]  # Dehnung XX
            nodal_counts[node_idx, 0:6] += 1.0

    # Average
    counts = nodal_counts.clone()
    counts[counts == 0] = 1.0
    nodal_avg = nodal_sums / counts

    # Displacements directly from u
    field_ux = u[:, 0]
    field_uy = u[:, 1]

    # Plot 3x3 layout
    fig = plt.figure(figsize=(12, 10))

    # 1: Unverformtes Netz (mit BC-Markern wie im Original)
    ax1 = plt.subplot(3, 3, 1)
    ax1.set_title("Unverformtes Netz")
    plot_mesh(ax1, solver.mesh, elems, x, plot_indices)
    # Dirichlet (grün), Neumann (rot)
    drlt_in = solver.mesh._drlt_in
    if drlt_in is not None and len(drlt_in) > 0:
        drlt_t = (
            torch.tensor(drlt_in) if not isinstance(drlt_in, torch.Tensor) else drlt_in
        )
        for i in range(drlt_t.size(0)):
            node_idx = int(drlt_t[i, 0].item())
            dof = int(drlt_t[i, 1].item())
            if dof == 0:
                ax1.plot(x[node_idx, 0], x[node_idx, 1], "g>", markersize=5)
            elif dof == 1:
                ax1.plot(x[node_idx, 0], x[node_idx, 1], "g^", markersize=5)
    neum_in = solver.mesh._neum_in
    if neum_in is not None and len(neum_in) > 0:
        neum_t = (
            torch.tensor(neum_in) if not isinstance(neum_in, torch.Tensor) else neum_in
        )
        for i in range(neum_t.size(0)):
            node_idx = int(neum_t[i, 0].item())
            dof = int(neum_t[i, 1].item())
            if dof == 0:
                ax1.plot(x[node_idx, 0], x[node_idx, 1], "r>", markersize=5)
            elif dof == 1:
                ax1.plot(x[node_idx, 0], x[node_idx, 1], "r^", markersize=5)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")

    # Helper to plot scalar nodal field (tricontourf on undeformed coords; deformed edges overlaid)
    def plot_scalar(ax, values, title, unit=None, scale=None):
        vals = values.clone()
        unit_str = ""
        if scale is not None:
            vals = vals * scale
        if unit is not None:
            unit_str = f" ({unit})"
        vmin = torch.min(vals)
        vmax = torch.max(vals)
        ax.set_title(f"{title}{unit_str}\nMax: {vmax:.2e}, Min: {vmin:.2e}")
        sc = ax.tricontourf(
            x[:, 0].detach().cpu(),
            x[:, 1].detach().cpu(),
            vals.detach().cpu(),
            levels=15,
            cmap=mpl.cm.jet,
        )
        plt.colorbar(sc, ax=ax)
        ax.axis("equal")

    # 2..7: fields
    ax2 = plt.subplot(3, 3, 2)
    plot_mesh(ax2, solver.mesh, elems, x_disped, plot_indices)
    plot_scalar(ax2, nodal_avg[:, 0] / 1e6, "Spannung XX", unit="MPa")

    ax3 = plt.subplot(3, 3, 3)
    plot_mesh(ax3, solver.mesh, elems, x_disped, plot_indices)
    plot_scalar(ax3, nodal_avg[:, 1] / 1e6, "Spannung YY", unit="MPa")

    ax4 = plt.subplot(3, 3, 4)
    plot_mesh(ax4, solver.mesh, elems, x_disped, plot_indices)
    plot_scalar(ax4, nodal_avg[:, 2] / 1e6, "Von-Mises-Spannung", unit="MPa")

    ax5 = plt.subplot(3, 3, 5)
    plot_mesh(ax5, solver.mesh, elems, x_disped, plot_indices)
    plot_scalar(ax5, field_ux * 1000.0, "Verschiebung u_x", unit="mm")

    ax6 = plt.subplot(3, 3, 6)
    plot_mesh(ax6, solver.mesh, elems, x_disped, plot_indices)
    plot_scalar(ax6, field_uy * 1000.0, "Verschiebung u_y", unit="mm")

    ax7 = plt.subplot(3, 3, 7)
    plot_mesh(ax7, solver.mesh, elems, x_disped, plot_indices)
    plot_scalar(ax7, nodal_avg[:, 5], "Dehnung XX")

    # 8: Cross-section at L/2 of sigma_xx
    ax8 = plt.subplot(3, 3, 8)
    mid_x = (x[:, 0].max() + x[:, 0].min()) / 2.0
    # pick nodes with x closest to mid_x
    x_coords = x[:, 0]
    idx_mid_x = torch.argmin(torch.abs(x_coords - mid_x))
    closest_x = x_coords[idx_mid_x]
    row_nodes = torch.where(x_coords == closest_x)[0]
    y_vals = x[row_nodes, 1]
    sigma_xx_row = nodal_avg[row_nodes, 0] / 1e6
    y_sorted, order = torch.sort(y_vals)
    sigma_sorted = sigma_xx_row[order]
    ax8.plot(
        sigma_sorted.detach().cpu().numpy(),
        y_sorted.detach().cpu().numpy(),
        "bo-",
        label="FEM Ergebnis",
    )
    # Analytical curve at mid-span (using exact same formula as original: M_z = 1000.0 * (L - x))
    y_analytical, sigma_analytical = analytical_sigma_xx_midspan(
        height=float(HEIGHT),
        width=float(WIDTH),
        length=float(LENGTH),
        force=float(FORCE),
        mid_point_x=float(closest_x),
    )
    ax8.plot(
        sigma_analytical.detach().cpu().numpy(),
        y_analytical.detach().cpu().numpy(),
        "r--",
        label="Analytische Lösung",
    )
    ax8.set_title(f"Biegespannung bei x = {float(closest_x):.3f} m")
    ax8.set_xlabel("Spannung XX (MPa)")
    ax8.set_ylabel("y-Koordinate (m)")
    ax8.grid(True)
    ax8.legend()

    fig.tight_layout()
