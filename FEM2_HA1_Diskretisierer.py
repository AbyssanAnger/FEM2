import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import time as timemodule

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)

disp_scaling = 50
toplot = True

tdm = 2


def analysis():
    E = 210e9  # E-Modul in Pa (210000 MPa)
    NU = 0.3

    NDF = 2
    NDM = 2
    global tdm
    NEN = 4
    FORCE = -1000.0
    LENGTH = 2.0  # m
    HEIGHT = 0.05  # m
    WIDTH = 0.05  # m (für Kraftberechnung, nicht für 2D-Netz)

    NX = 120  # Anzahl der Elemente in Längsrichtung
    NY = 10  # Anzahl der Elemente in Höhenrichtung

    # Erzeuge Knoten
    x_coords = torch.linspace(0, LENGTH, NX + 1)
    y_coords = torch.linspace(0, HEIGHT, NY + 1)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
    x = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1)

    # Erzeuge Elemente (Konnektivität)
    elements_list = []
    for j in range(NY):
        for i in range(NX):
            n1 = i * (NY + 1) + j  # Unten links
            n2 = (i + 1) * (NY + 1) + j  # Unten rechts
            n3 = (i + 1) * (NY + 1) + j + 1  # Oben rechts
            n4 = i * (NY + 1) + j + 1  # Oben links
            elements_list.append([n1, n2, n3, n4])
    elems = torch.tensor(elements_list, dtype=torch.long)

    # Einspannung am linken Rand (x=0)
    left_nodes = torch.where(x[:, 0] == 0)[0]
    drlt_list = [[[node_idx, 0, 0], [node_idx, 1, 0]] for node_idx in left_nodes]
    drlt = torch.tensor([item for sublist in drlt_list for item in sublist])

    # Querkraft am rechten Rand (x=LENGTH)
    total_force = FORCE / WIDTH  # Kraft pro Meter Breite [N/m]
    right_nodes = torch.where(x[:, 0] == LENGTH)[0]
    force_per_node = total_force / len(right_nodes)
    neum_list = [[node_idx, 1, force_per_node] for node_idx in right_nodes]
    neum = torch.tensor(neum_list)

    # Gauss quadrature
    nqp = 4

    # Materialtensor für ebenen Spannungszustand (plane stress)
    C4 = torch.zeros(2, 2, 2, 2)
    factor = E / (1 - NU**2)
    C4[0, 0, 0, 0] = factor * 1
    C4[1, 1, 1, 1] = factor * 1
    C4[0, 0, 1, 1] = factor * NU
    C4[1, 1, 0, 0] = factor * NU

    shear_factor = E / (2 * (1 + NU))
    C4[0, 1, 0, 1] = C4[1, 0, 0, 1] = C4[0, 1, 1, 0] = C4[1, 0, 1, 0] = shear_factor

    nnp = x.size()[0]
    print("nnp: ", nnp)
    nel = elems.size()[0]

    drlt_mask = torch.zeros(nnp * NDF, 1)

    drlt_vals = torch.zeros(nnp * NDF, 1)
    for i in range(drlt.size()[0]):
        drlt_mask[int(drlt[i, 0]) * NDF + int(drlt[i, 1]), 0] = 1
        drlt_vals[int(drlt[i, 0]) * NDF + int(drlt[i, 1]), 0] = drlt[i, 2]
    free_mask = torch.ones(nnp * NDF, 1) - drlt_mask
    drltDofs = torch.nonzero(drlt_mask)
    print("drltDofs", drltDofs)

    drlt_matrix = 1e22 * torch.diag(drlt_mask[:, 0], 0)

    neum_vals = torch.zeros(nnp * NDF, 1)
    for i in range(neum.size()[0]):
        neum_vals[int(neum[i, 0]) * NDF + int(neum[i, 1]), 0] = neum[i, 2]

    qpt = torch.zeros(nqp, NDM)
    a = math.sqrt(3) / 3
    qpt[0, 0] = -a
    qpt[0, 1] = -a
    qpt[1, 0] = a
    qpt[1, 1] = -a
    qpt[2, 0] = -a
    qpt[2, 1] = a
    qpt[3, 0] = a
    qpt[3, 1] = a

    w8 = torch.ones(nqp, 1)

    # Masterelement - Shape function
    masterelem_N = torch.zeros(nqp, 4)
    masterelem_gamma = torch.zeros(nqp, 4, 2)

    for q in range(nqp):
        xi = qpt[q, :]
        masterelem_N[q, 0] = 0.25 * (1 - xi[0]) * (1 - xi[1])
        masterelem_N[q, 1] = 0.25 * (1 + xi[0]) * (1 - xi[1])
        masterelem_N[q, 2] = 0.25 * (1 + xi[0]) * (1 + xi[1])
        masterelem_N[q, 3] = 0.25 * (1 - xi[0]) * (1 + xi[1])

        masterelem_gamma[q, 0, 0] = -0.25 * (1 - xi[1])
        masterelem_gamma[q, 0, 1] = -0.25 * (1 - xi[0])
        masterelem_gamma[q, 1, 0] = 0.25 * (1 - xi[1])
        masterelem_gamma[q, 1, 1] = -0.25 * (1 + xi[0])
        masterelem_gamma[q, 2, 0] = 0.25 * (1 + xi[1])
        masterelem_gamma[q, 2, 1] = 0.25 * (1 + xi[0])
        masterelem_gamma[q, 3, 0] = -0.25 * (1 + xi[1])
        masterelem_gamma[q, 3, 1] = 0.25 * (1 - xi[0])

    indices = torch.tensor([0, 1, 2, 3, 0])

    u = torch.zeros(NDF * nnp, 1)
    print("x", x)

    edof = torch.zeros(nel, NDF, NEN, dtype=int)
    gdof = torch.zeros(nel, NDF * NEN, dtype=int)
    for el in range(nel):
        for ien in range(NEN):
            for idf in range(NDF):
                edof[el, idf, ien] = NDF * elems[el, ien] + idf
        gdof[el, :] = edof[el, :, :].t().reshape(NDF * NEN)

    finte = torch.zeros(NEN * NDF, 1)
    fvole = torch.zeros(NEN * NDF, 1)

    Ke = torch.zeros(NEN * NDF, NEN * NDF)
    K = torch.zeros(nnp * NDF, nnp * NDF)

    h = torch.zeros(tdm, tdm)

    G_A = torch.zeros(1, NDM)

    fsur = neum_vals

    fint = torch.zeros(NDF * nnp, 1)
    fvol = torch.zeros(NDF * nnp, 1)

    for el in range(nel):
        xe = torch.zeros(NDM, NEN)
        for idm in range(NDM):
            xe[idm, :] = x[elems[el, :], idm]

        ue = torch.squeeze(u[edof[el, 0:NDM, :]])

        Ke.zero_()
        finte.zero_()
        for q in range(nqp):
            gamma = masterelem_gamma[q]

            Je = xe.mm(gamma)
            detJe = torch.det(Je)

            if detJe <= 0:
                print("Error: detJe <= 0")

            dv = detJe * w8[q]
            invJe = torch.inverse(Je)

            G = gamma.mm(invJe)

            h[0:NDM, 0:NDM] = ue.mm(G)
            eps_2d = 0.5 * (h + h.t())[0:NDM, 0:NDM]
            stre_2d = torch.tensordot(C4, eps_2d, dims=2)

            for A in range(NEN):
                G_A[0, :] = G[A, :]

                fintA = dv * (G_A.mm(stre_2d))
                finte[A * NDF : A * NDF + NDM] += fintA.t()

                for B in range(NEN):
                    KAB = torch.tensordot(
                        G[A, :],
                        (
                            torch.tensordot(
                                C4[0:tdm, 0:tdm, 0:tdm, 0:tdm], G[B, :], [[3], [0]]
                            )
                        ),
                        [[0], [0]],
                    )
                    Ke[A * NDF : A * NDF + NDM, B * NDF : B * NDF + NDM] += dv * KAB

        fint[gdof[el, :]] += finte
        fvol[gdof[el, :]] += fvole
        for i in range(gdof.shape[1]):
            K[gdof[el, i], gdof[el, :]] += Ke[i, :]

    ############################end of Element#####################################

    rsd = free_mask.mul(fsur - fint)

    K_tilde = K + drlt_matrix

    du = torch.linalg.solve(K_tilde, rsd)
    u += du
    u = free_mask.mul(u) + drlt_mask.mul(drlt_vals)
    print("u: ", u)

    u_reshaped = torch.reshape(u, (-1, 2))

    x_disped = x + disp_scaling * u_reshaped

    plot_titles = [
        "Spannung XX",
        "Spannung YY",
        "Von-Mises-Spannung",
        "Verschiebung u_x",
        "Verschiebung u_y",
        "Dehnung XX",
    ]

    nodal_stresses_sum = torch.zeros(nnp, len(plot_titles))
    nodal_contribution_count = torch.zeros(nnp, len(plot_titles))
    N_at_gps = torch.vstack([masterelem_N[q] for q in range(nqp)])
    try:
        extrapolation_matrix = torch.inverse(N_at_gps)
    except torch.linalg.LinAlgError:
        extrapolation_matrix = torch.linalg.pinv(N_at_gps)

    # Felder sammeln: s_xx, s_yy, s_vm, eps_xx
    for e in range(nel):
        xe = torch.zeros(NDM, NEN)
        for idm in range(NDM):
            xe[idm, :] = x[elems[e, :], idm]
        ue = torch.squeeze(u[edof[e, 0:NDM, :]])

        s_xx_gp = torch.zeros(nqp)
        s_yy_gp = torch.zeros(nqp)
        s_vm_gp = torch.zeros(nqp)
        eps_xx_gp = torch.zeros(nqp)

        for q in range(nqp):
            gamma = masterelem_gamma[q]
            Je = xe.mm(gamma)
            detJe = torch.det(Je)
            if detJe <= 0:
                continue
            invJe = torch.inverse(Je)
            G = gamma.mm(invJe)

            h_2d = ue.mm(G)
            eps_2d = 0.5 * (h_2d + h_2d.t())
            stre_2d = torch.tensordot(C4, eps_2d, dims=2)

            s11 = stre_2d[0, 0]
            s22 = stre_2d[1, 1]
            s12 = stre_2d[0, 1]
            s33 = 0.0
            sigma_v = torch.sqrt(
                0.5
                * (
                    (s11 - s22) ** 2
                    + (s22 - s33) ** 2
                    + (s33 - s11) ** 2
                    + 6 * (s12**2)
                )
            )

            s_xx_gp[q] = s11
            s_yy_gp[q] = s22
            s_vm_gp[q] = sigma_v
            eps_xx_gp[q] = eps_2d[0, 0]

        nodal_vals_elem = torch.stack(
            [
                extrapolation_matrix.mv(s_xx_gp),
                extrapolation_matrix.mv(s_yy_gp),
                extrapolation_matrix.mv(s_vm_gp),
                extrapolation_matrix.mv(eps_xx_gp),
            ]
        )  # (4, NEN)

        element_nodes = elems[e, :]
        nodal_stresses_sum[element_nodes, 0] += nodal_vals_elem[0]
        nodal_stresses_sum[element_nodes, 1] += nodal_vals_elem[1]
        nodal_stresses_sum[element_nodes, 2] += nodal_vals_elem[2]
        nodal_stresses_sum[element_nodes, 5] += nodal_vals_elem[3]  # Dehnung XX
        nodal_contribution_count[element_nodes, 0] += 1
        nodal_contribution_count[element_nodes, 1] += 1
        nodal_contribution_count[element_nodes, 2] += 1
        nodal_contribution_count[element_nodes, 5] += 1

    # Mittelung
    counts = nodal_contribution_count.clone()
    counts[counts == 0] = 1
    nodal_avg = nodal_stresses_sum / counts

    # Hilfsfunktionen fr saubere Plots
    def plot_mesh(ax, coords, connectivity, idx_seq):
        for e in range(nel):
            els = torch.index_select(connectivity[e, :], 0, idx_seq)
            ax.plot(coords[els, 0], coords[els, 1], "k-", linewidth=0.5)
        ax.plot(coords[:, 0], coords[:, 1], "ko", markersize=2)
        ax.axis("equal")

    def plot_scalar(ax, values, title, unit=None, scale=None):
        vals = values.clone()
        unit_str = ""
        if scale is not None:
            vals = vals * scale
        if unit is not None:
            unit_str = f" ({unit})"
        vmin = torch.min(vals)
        vmax = torch.max(vals)
        ax.set_title(f"{title}{unit_str}\nMax: {vmax:.3e}, Min: {vmin:.3e}")
        sc = ax.tricontourf(
            x[:, 0].detach(),
            x[:, 1].detach(),
            vals.detach(),
            levels=15,
            cmap=mpl.cm.jet,
        )
        plt.colorbar(sc, ax=ax)
        ax.axis("equal")

    plt.figure(figsize=(12, 10))

    # 1: Unverformtes Netz mit BC-Markern
    ax1 = plt.subplot(3, 3, 1)
    ax1.set_title("Unverformtes Netz")
    plot_mesh(ax1, x, elems, indices)
    for drlt_bc in drlt:
        node_idx = int(drlt_bc[0])
        if drlt_bc[1] == 0:
            ax1.plot(x[node_idx, 0], x[node_idx, 1], "g>", markersize=5)
        if drlt_bc[1] == 1:
            ax1.plot(x[node_idx, 0], x[node_idx, 1], "g^", markersize=5)
    for neum_bc in neum:
        node_idx = int(neum_bc[0])
        if neum_bc[1] == 0:
            ax1.plot(x[node_idx, 0], x[node_idx, 1], "r>", markersize=5)
        if neum_bc[1] == 1:
            ax1.plot(x[node_idx, 0], x[node_idx, 1], "r^", markersize=5)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")

    # Felder 2..7 mit berlagerter verformter Kante
    ax2 = plt.subplot(3, 3, 2)
    plot_mesh(ax2, x_disped, elems, indices)
    plot_scalar(ax2, (nodal_avg[:, 0] / 1e6), "Spannung XX", unit="MPa")

    ax3 = plt.subplot(3, 3, 3)
    plot_mesh(ax3, x_disped, elems, indices)
    plot_scalar(ax3, (nodal_avg[:, 1] / 1e6), "Spannung YY", unit="MPa")

    ax4 = plt.subplot(3, 3, 4)
    plot_mesh(ax4, x_disped, elems, indices)
    plot_scalar(ax4, (nodal_avg[:, 2] / 1e6), "Von-Mises-Spannung", unit="MPa")

    ax5 = plt.subplot(3, 3, 5)
    plot_mesh(ax5, x_disped, elems, indices)
    plot_scalar(ax5, u_reshaped[:, 0] * 1000.0, "Verschiebung u_x", unit="mm")

    ax6 = plt.subplot(3, 3, 6)
    plot_mesh(ax6, x_disped, elems, indices)
    plot_scalar(ax6, u_reshaped[:, 1] * 1000.0, "Verschiebung u_y", unit="mm")

    ax7 = plt.subplot(3, 3, 7)
    plot_mesh(ax7, x_disped, elems, indices)
    plot_scalar(ax7, nodal_avg[:, 5], "Dehnung XX")

    ax_cross_section = plt.subplot(3, 3, 8)
    mid_point_x = LENGTH / 2.0
    closest_x_value = x_coords[torch.argmin(torch.abs(x_coords - mid_point_x))]
    mid_nodes_indices = torch.where(x[:, 0] == closest_x_value)[0]

    y_vals_mid = x[mid_nodes_indices, 1]
    stress_xx_counts = nodal_contribution_count[:, 0].clone()
    stress_xx_counts[stress_xx_counts == 0] = 1
    stress_xx_nodal = nodal_stresses_sum[:, 0] / stress_xx_counts
    stress_xx_mid = stress_xx_nodal[mid_nodes_indices] / 1e6

    y_sorted, indices_sorted = torch.sort(y_vals_mid)
    stress_sorted = stress_xx_mid[indices_sorted]

    I_z = (WIDTH * HEIGHT**3) / 12
    M_z = (1000.0) * (LENGTH - mid_point_x)
    y_analytical = torch.linspace(0, HEIGHT, 100)
    sigma_analytical = (M_z / I_z * (y_analytical - HEIGHT / 2)) / 1e6

    ax_cross_section.plot(
        stress_sorted.numpy(), y_sorted.numpy(), "bo-", label="FEM Ergebnis"
    )
    ax_cross_section.plot(
        sigma_analytical.numpy(),
        y_analytical.numpy(),
        "r--",
        label="Analytische Lösung",
    )
    ax_cross_section.set_title(f"Biegespannung bei x = {mid_point_x:.3f} m")
    ax_cross_section.set_xlabel("Spannung XX (MPa)")
    ax_cross_section.set_ylabel("y-Koordinate (m)")
    ax_cross_section.grid(True)
    ax_cross_section.legend()

    plt.tight_layout()


if __name__ == "__main__":

    start_perfcount = timemodule.perf_counter()
    analysis()
    end_perfcount = timemodule.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end_perfcount - start_perfcount)))

    if toplot:
        plt.show()
