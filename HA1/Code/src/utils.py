# utils.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from config import LENGTH, HEIGHT, WIDTH, FORCE, TOTAL_FORCE
from src.analytics import analytical_sigma_xx_midspan


def plot_mesh(ax, mesh, elems, x, indices):
    for e in range(mesh.nel):
        els = elems[e, :][indices]
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
        extrap = torch.inverse(N_at_gps)
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
    plot_mesh(ax1, solver.mesh, elems, x, solver.indices)
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
    plot_mesh(ax2, solver.mesh, elems, x_disped, solver.indices)
    plot_scalar(ax2, nodal_avg[:, 0] / 1e6, "Spannung XX", unit="MPa")

    ax3 = plt.subplot(3, 3, 3)
    plot_mesh(ax3, solver.mesh, elems, x_disped, solver.indices)
    plot_scalar(ax3, nodal_avg[:, 1] / 1e6, "Spannung YY", unit="MPa")

    ax4 = plt.subplot(3, 3, 4)
    plot_mesh(ax4, solver.mesh, elems, x_disped, solver.indices)
    plot_scalar(ax4, nodal_avg[:, 2] / 1e6, "Von-Mises-Spannung", unit="MPa")

    ax5 = plt.subplot(3, 3, 5)
    plot_mesh(ax5, solver.mesh, elems, x_disped, solver.indices)
    plot_scalar(ax5, field_ux * 1000.0, "Verschiebung u_x", unit="mm")

    ax6 = plt.subplot(3, 3, 6)
    plot_mesh(ax6, solver.mesh, elems, x_disped, solver.indices)
    plot_scalar(ax6, field_uy * 1000.0, "Verschiebung u_y", unit="mm")

    ax7 = plt.subplot(3, 3, 7)
    plot_mesh(ax7, solver.mesh, elems, x_disped, solver.indices)
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
