import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import time as timemodule

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

disp_scaling = 50
toplot = True

tdm = 2


def analysis():
    E = 210e9  # E-Modul in Pa (210000 MPa)
    nu = 0.3

    ndf = 2
    ndm = 2
    global tdm
    nen = 4

    # --- Geometrie: Prozedurale Erzeugung eines Balkennetzes ---
    length = 2.0  # m
    height = 0.05  # m
    width = 0.05  # m (für Kraftberechnung, nicht für 2D-Netz)

    nx = 20  # Anzahl der Elemente in Längsrichtung
    ny = 2  # Anzahl der Elemente in Höhenrichtung
    nx = 120  # Anzahl der Elemente in Längsrichtung
    ny = 10  # Anzahl der Elemente in Höhenrichtung (wichtig für genauen Verlauf)

    # Erzeuge Knoten
    x_coords = torch.linspace(0, length, nx + 1)
    y_coords = torch.linspace(0, height, ny + 1)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
    x = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1)

    # Erzeuge Elemente (Konnektivität)
    elements_list = []
    for j in range(ny):
        for i in range(nx):
            n1 = i * (ny + 1) + j  # Unten links
            n2 = (i + 1) * (ny + 1) + j  # Unten rechts
            n3 = (i + 1) * (ny + 1) + j + 1  # Oben rechts
            n4 = i * (ny + 1) + j + 1  # Oben links
            elements_list.append([n1, n2, n3, n4])
    elems = torch.tensor(elements_list, dtype=torch.long)

    # Einspannung am linken Rand (x=0)
    left_nodes = torch.where(x[:, 0] == 0)[0]
    drlt_list = [[[node_idx, 0, 0], [node_idx, 1, 0]] for node_idx in left_nodes]
    drlt = torch.tensor([item for sublist in drlt_list for item in sublist])

    # Querkraft am rechten Rand (x=length)
    total_force = -1000.0 / width  # Kraft pro Meter Breite [N/m]
    right_nodes = torch.where(x[:, 0] == length)[0]
    force_per_node = total_force / len(right_nodes)
    neum_list = [[node_idx, 1, force_per_node] for node_idx in right_nodes]
    neum = torch.tensor(neum_list)

    # Gauss quadrature
    nqp = 4

    ############ Identity tensors ###########
    ei = torch.eye(ndm, ndm)

    I = torch.eye(3, 3)

    blk = E / (3 * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    # C4 = blk * I4 + 2 * mu * I4dev
    lame = blk - 2 / 3 * mu

    # Korrekter Materialtensor für ebenen Spannungszustand (plane stress)
    C4 = torch.zeros(2, 2, 2, 2)
    factor = E / (1 - nu**2)
    C4[0, 0, 0, 0] = factor * 1
    C4[1, 1, 1, 1] = factor * 1
    C4[0, 0, 1, 1] = factor * nu
    C4[1, 1, 0, 0] = factor * nu

    shear_factor = E / (2 * (1 + nu))  # Schubmodul G
    C4[0, 1, 0, 1] = C4[1, 0, 0, 1] = C4[0, 1, 1, 0] = C4[1, 0, 1, 0] = shear_factor

    ############ Preprocessing ###############
    nnp = x.size()[0]
    print("nnp: ", nnp)
    nel = elems.size()[0]

    drlt_mask = torch.zeros(nnp * ndf, 1)

    drlt_vals = torch.zeros(nnp * ndf, 1)
    for i in range(drlt.size()[0]):
        drlt_mask[int(drlt[i, 0]) * ndf + int(drlt[i, 1]), 0] = 1
        drlt_vals[int(drlt[i, 0]) * ndf + int(drlt[i, 1]), 0] = drlt[i, 2]
    free_mask = torch.ones(nnp * ndf, 1) - drlt_mask
    drltDofs = torch.nonzero(drlt_mask)
    print("drltDofs", drltDofs)

    drlt_matrix = 1e22 * torch.diag(
        drlt_mask[:, 0], 0
    )  # Große Zahl auf den Diagonalen der festgehaltenen Freiheitsgrade (Penalty-Methode Steifigkeitsmethode)

    neum_vals = torch.zeros(nnp * ndf, 1)
    for i in range(neum.size()[0]):
        neum_vals[int(neum[i, 0]) * ndf + int(neum[i, 1]), 0] = neum[i, 2]

    qpt = torch.zeros(nqp, ndm)
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

    # Plot des unverformten Netzes
    plt.subplot(3, 3, 1)
    plt.title("Unverformtes Netz")
    for e in range(nel):
        els = torch.index_select(elems[e, :], 0, indices)
        plt.plot(x[els, 0], x[els, 1], "k-", linewidth=0.5)
    plt.plot(x[:, 0], x[:, 1], "ko", markersize=2)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # Randbedingungen einzeichnen
    for drlt_bc in drlt:
        node_idx = int(drlt_bc[0])
        if drlt_bc[1] == 0:  # x-Richtung
            plt.plot(x[node_idx, 0], x[node_idx, 1], "g>", markersize=5)
        if drlt_bc[1] == 1:  # y-Richtung
            plt.plot(x[node_idx, 0], x[node_idx, 1], "g^", markersize=5)

    for neum_bc in neum:
        node_idx = int(neum_bc[0])
        if neum_bc[1] == 0:  # x-Richtung
            plt.plot(x[node_idx, 0], x[node_idx, 1], "r>", markersize=5)
        if neum_bc[1] == 1:  # y-Richtung
            plt.plot(x[node_idx, 0], x[node_idx, 1], "r^", markersize=5)

    ############## Analysis ###############
    u = torch.zeros(ndf * nnp, 1)
    print("x", x)

    edof = torch.zeros(nel, ndf, nen, dtype=int)
    gdof = torch.zeros(nel, ndf * nen, dtype=int)
    for el in range(nel):
        for ien in range(nen):
            for idf in range(ndf):
                edof[el, idf, ien] = ndf * elems[el, ien] + idf
        gdof[el, :] = edof[el, :, :].t().reshape(ndf * nen)

    finte = torch.zeros(nen * ndf, 1)
    fvole = torch.zeros(nen * ndf, 1)

    Ke = torch.zeros(nen * ndf, nen * ndf)
    K = torch.zeros(nnp * ndf, nnp * ndf)

    h = torch.zeros(tdm, tdm)

    G_A = torch.zeros(1, ndm)

    fsur = neum_vals

    fint = torch.zeros(ndf * nnp, 1)
    fvol = torch.zeros(ndf * nnp, 1)

    ############################Element#####################################
    for el in range(nel):
        xe = torch.zeros(ndm, nen)
        for idm in range(ndm):
            xe[idm, :] = x[elems[el, :], idm]

        ue = torch.squeeze(u[edof[el, 0:ndm, :]])

        Ke.zero_()
        for q in range(nqp):
            N = masterelem_N[q]
            gamma = masterelem_gamma[q]

            Je = xe.mm(gamma)
            detJe = torch.det(Je)

            if detJe <= 0:
                print("Error: detJe <= 0")

            dv = detJe * w8[q]
            invJe = torch.inverse(Je)

            G = gamma.mm(invJe)

            h[0:ndm, 0:ndm] = ue.mm(G)
            eps_2d = 0.5 * (h + h.t())[0:ndm, 0:ndm]
            stre_2d = torch.tensordot(C4, eps_2d, dims=2)

            for A in range(nen):
                G_A[0, :] = G[A, :]

                fintA = dv * (G_A.mm(stre_2d))
                finte[A * ndf : A * ndf + ndm] += fintA.t()

                for B in range(nen):
                    KAB = torch.tensordot(
                        G[A, :],
                        (
                            torch.tensordot(
                                C4[0:tdm, 0:tdm, 0:tdm, 0:tdm], G[B, :], [[3], [0]]
                            )
                        ),
                        [[0], [0]],
                    )
                    Ke[A * ndf : A * ndf + ndm, B * ndf : B * ndf + ndm] += dv * KAB

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

    fext = K * u
    frea = fext - fvol - fsur

    ###### Post-processing/ plots ########
    u_reshaped = torch.reshape(u, (-1, 2))

    x_disped = x + disp_scaling * u_reshaped

    voigt = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]])
    # --- ANPASSUNG: Plot-Titel ändern, um Verschiebungen statt Schubspannungen anzuzeigen ---
    plot_titles = [
        "Spannung XX",
        "Spannung YY",
        "Von-Mises-Spannung",
        "Verschiebung u_x",
        "Verschiebung u_y",
        "Dehnung XX",
    ]
    ei = torch.eye(3, 3)
    # Die Größe von plotdata ist nicht mehr kritisch, da wir Knotenergebnisse plotten
    plotdata = torch.zeros(len(plot_titles), nel * nqp, 3)

    # --- NEU: Initialisierung für Knoten-Spannungs-Extrapolation ---
    # Tensor zum Speichern der summierten Spannungswerte an jedem Knoten
    nodal_stresses_sum = torch.zeros(nnp, len(plot_titles))
    # Tensor zum Zählen, wie viele Elemente zu jedem Knoten beitragen (für die Mittelung)
    nodal_contribution_count = torch.zeros(nnp, len(plot_titles))

    for i in range(len(plot_titles)):
        ax = plt.subplot(3, 3, i + 2)  # Subplots 2, 3, 4, 5, 6, 7
        for e in range(nel):
            els = torch.index_select(elems[e, :], 0, indices)
            ax.plot(x_disped[els, 0], x_disped[els, 1], "b-", linewidth=0.5)

            xe = torch.zeros(ndm, nen)
            for idm in range(ndm):
                xe[idm, :] = x[elems[e, :], idm]
            ue = torch.squeeze(u[edof[e, 0:ndm, :]])

            # Temporärer Speicher für Spannungen an Gauß-Punkten eines Elements
            stresses_at_gauss_points = torch.zeros(nqp)

            for q in range(nqp):
                xgauss = torch.mv(
                    torch.transpose(x_disped[elems[e, :]], 0, 1), masterelem_N[q]
                )

                gamma = masterelem_gamma[q]
                Je = xe.mm(gamma)
                invJe = torch.inverse(Je)
                G = gamma.mm(invJe)

                h = torch.zeros(3, 3)
                h_2d = ue.mm(G)
                eps_2d = 0.5 * (h_2d + h_2d.t())

                stre_2d = torch.tensordot(C4, eps_2d, dims=2)

                stre = torch.zeros(3, 3)
                stre[0:2, 0:2] = stre_2d
                eps = torch.zeros(3, 3)
                eps[0:2, 0:2] = eps_2d
                eps[2, 2] = -nu / (1 - nu) * (eps_2d[0, 0] + eps_2d[1, 1])

                s11, s22, s12 = stre[0, 0], stre[1, 1], stre[0, 1]
                s33 = 0.0  # Annahme für ebenen Spannungszustand
                sigma_v = torch.sqrt(
                    0.5
                    * (
                        (s11 - s22) ** 2
                        + (s22 - s33) ** 2
                        + (s33 - s11) ** 2
                        + 6 * (s12**2)
                    )
                )

                # --- ANPASSUNG: Logik zur Auswahl der Plot-Werte ---
                title = plot_titles[i]
                if title == "Spannung XX":
                    stre_val = stre[0, 0]
                elif title == "Spannung YY":
                    stre_val = stre[1, 1]
                elif title == "Von-Mises-Spannung":
                    stre_val = sigma_v
                elif title == "Dehnung XX":
                    stre_val = eps[0, 0]
                else:
                    # Für Verschiebungen ist keine Gauß-Punkt-Berechnung nötig
                    stre_val = 0

                if "Spannung" in title or "Dehnung" in title:
                    stresses_at_gauss_points[q] = stre_val

            # Extrapolation von Gauß-Punkten zu den Knoten des Elements
            N_at_gps = torch.vstack([masterelem_N[q] for q in range(nqp)])
            try:
                extrapolation_matrix = torch.inverse(N_at_gps)
            except torch.linalg.LinAlgError:
                # Fallback auf Pseudo-Inverse, falls Matrix singulär ist
                extrapolation_matrix = torch.linalg.pinv(N_at_gps)

            # Nur für Spannungs- und Dehnungswerte extrapolieren
            if "Spannung" in plot_titles[i] or "Dehnung" in plot_titles[i]:
                nodal_stresses_element = extrapolation_matrix.mv(
                    stresses_at_gauss_points
                )
                # Addiere die extrapolierten Werte zu den globalen Summen-Tensoren
                element_nodes = elems[e, :]
                nodal_stresses_sum[element_nodes, i] += nodal_stresses_element
                nodal_contribution_count[element_nodes, i] += 1

        # Mittelung der Knotenspannungen nach der Schleife über alle Elemente
        title = plot_titles[i]
        if "Spannung" in title or "Dehnung" in title:
            # Vermeide Division durch Null für Knoten, die nicht verwendet werden
            valid_counts = nodal_contribution_count[:, i].clone()
            valid_counts[valid_counts == 0] = 1
            averaged_nodal_stresses = nodal_stresses_sum[:, i] / valid_counts
            plot_values_nodal = averaged_nodal_stresses.clone()
        elif title == "Verschiebung u_x":
            plot_values_nodal = u_reshaped[:, 0]  # Direkter Zugriff auf Knotenergebnis
        elif title == "Verschiebung u_y":
            plot_values_nodal = u_reshaped[:, 1]  # Direkter Zugriff auf Knotenergebnis

        # --- ANPASSUNG: Verwende die gemittelten Knotenspannungen für den Plot ---
        if "Spannung" in title:
            plot_values_nodal /= 1e6  # Umrechnung in MPa
            unit_str = " (MPa)"
        elif "Verschiebung" in title:
            plot_values_nodal *= 1000  # Umrechnung in mm
            unit_str = " (mm)"
        else:
            unit_str = ""  # Dehnung ist einheitenlos

        min_val_nodal = torch.min(plot_values_nodal)
        max_val_nodal = torch.max(plot_values_nodal)

        ax.set_title(
            f"{plot_titles[i]}{unit_str}\nMax: {max_val_nodal:.2e}, Min: {min_val_nodal:.2e}"
        )

        # tricontourf für eine glatte Darstellung der Knotenergebnisse
        scatter = ax.tricontourf(
            x[:, 0], x[:, 1], plot_values_nodal, levels=15, cmap=mpl.cm.jet
        )

        plt.colorbar(scatter, ax=ax)

        ax.axis("equal")

    # --- NEU: Letzter Plot - Spannungsverlauf bei L/2 ---
    ax_cross_section = plt.subplot(3, 3, 8)  # Plot an Position 8

    # Finde Knoten in der Mitte des Balkens (x = L/2)
    mid_point_x = length / 2.0
    # Finde den x-Wert im Gitter, der der Mitte am nächsten kommt
    closest_x_value = x_coords[torch.argmin(torch.abs(x_coords - mid_point_x))]
    mid_nodes_indices = torch.where(x[:, 0] == closest_x_value)[0]

    # Extrahiere y-Koordinaten und Spannung XX für diese Knoten
    y_vals_mid = x[mid_nodes_indices, 1]
    # Spannung XX ist der erste Plot (Index 0)
    stress_xx_counts = nodal_contribution_count[:, 0].clone()
    stress_xx_counts[stress_xx_counts == 0] = 1
    stress_xx_nodal = nodal_stresses_sum[:, 0] / stress_xx_counts

    stress_xx_mid = stress_xx_nodal[mid_nodes_indices] / 1e6  # in MPa

    # Sortiere die Werte nach y-Koordinate für einen sauberen Plot
    y_sorted, indices_sorted = torch.sort(y_vals_mid)
    stress_sorted = stress_xx_mid[indices_sorted]

    # Analytische Lösung bei L/2
    I_z = (width * height**3) / 12
    M_z = (1000.0) * (length - mid_point_x)  # F * (L-x)
    y_analytical = torch.linspace(0, height, 100)
    sigma_analytical = (M_z / I_z * (y_analytical - height / 2)) / 1e6  # in MPa

    ax_cross_section.plot(
        stress_sorted.numpy(), y_sorted.numpy(), "bo-", label="FEM Ergebnis"
    )
    ax_cross_section.plot(
        sigma_analytical.numpy(),
        y_analytical.numpy(),
        "r--",
        label="Analytische Lösung",
    )
    ax_cross_section.set_title(f"Biegespannung bei x = {mid_point_x:.1f} m")
    ax_cross_section.set_xlabel("Spannung XX (MPa)")
    ax_cross_section.set_ylabel("y-Koordinate (m)")
    ax_cross_section.grid(True)
    ax_cross_section.legend()


if __name__ == "__main__":

    start_perfcount = timemodule.perf_counter()
    analysis()
    end_perfcount = timemodule.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end_perfcount - start_perfcount)))

    if toplot:
        plt.show()
