import numpy as np
import matplotlib.pyplot as plt


def create_geometry(l, b, h, nx=2, ny=2):
    """
    Generiert die Geometrie eines rechteckigen 2D-FEM-Meshs für einen Balken.

    Parameter:
    - l: Länge in x-Richtung
    - b: Breite in y-Richtung
    - h: Höhe/Dicke (derzeit für 2D ignoriert, für 3D erweiterbar)
    - nx: Anzahl Elemente in x-Richtung (Standard: 2)
    - ny: Anzahl Elemente in y-Richtung (Standard: 2)

    Returns:
    - coords: Knoten-Koordinaten (N x 2)
    - elems: Element-Konnektivitäten (M x 4, für quadratische Elemente)
    """
    # Knoten generieren: (nx+1) x (ny+1) Gitter
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    x = np.linspace(0, l, n_nodes_x)
    y = np.linspace(0, b, n_nodes_y)
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack(
        (X.ravel(), Y.ravel())
    )  # Flach: [y0x0, y0x1, ..., yNy xNx]

    # Elemente generieren: Für jedes Element (i,j): Knoten-IDs [y_i x_j, y_i x_{j+1}, y_{i+1} x_{j+1}, y_{i+1} x_j]
    elems = []
    for i in range(ny):
        for j in range(nx):
            n1 = i * n_nodes_x + j  # Unten links
            n2 = i * n_nodes_x + (j + 1)  # Unten rechts
            n3 = (i + 1) * n_nodes_x + (j + 1)  # Oben rechts
            n4 = (i + 1) * n_nodes_x + j  # Oben links
            elems.append([n1, n2, n3, n4])
    elems = np.array(elems)

    return coords, elems


def create_boundary_conditions(l, b, nx=2, ny=2, force=50000):
    """
    Generiert die Randbedingungen für das rechteckige 2D-FEM-Mesh eines Balkens.

    Parameter:
    - l: Länge in x-Richtung
    - b: Breite in y-Richtung
    - nx: Anzahl Elemente in x-Richtung (Standard: 2)
    - ny: Anzahl Elemente in y-Richtung (Standard: 2)
    - force: Basislast (Standard: 50000)

    Returns:
    - drlt: Dirichlet-BCs [Knoten-ID, DOF, Wert] (linke Seite fixiert in x)
    - neum: Neumann-BCs [Knoten-ID, DOF, Wert] (rechte Seite Last in y)
    """
    # Anzahl Knoten berechnen (benötigt für Knoten-IDs)
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1

    # Dirichlet-BCs: Alle Knoten an linker Seite (x=0, d.h. j=0) fixiert in DOF=0 (x-Versatz=0)
    drlt = []
    left_nodes = np.arange(
        0, n_nodes_y * n_nodes_x, n_nodes_x
    )  # Knoten-IDs: 0, n_nodes_x, 2*n_nodes_x, ...
    for node_id in left_nodes:
        drlt.append([node_id, 0, 0])  # DOF=0, Wert=0
    drlt = np.array(drlt)

    # Neumann-BCs: Last in y-Richtung (DOF=1) an rechter Seite (j=nx), Werte proportional verteilt
    # Originalwerte: 5e6, 1e7, 5e6 – hier skaliert und gleichmäßig über ny+1 Knoten verteilt (z.B. linear)
    right_nodes = np.arange(nx, n_nodes_y * n_nodes_x, n_nodes_x)  # Knoten-IDs an x=l
    # Beispiel: Lineare Verteilung der Last (von niedrig zu hoch zu niedrig, skaliert mit b)
    base_load = force  # MPa
    loads = (
        base_load * np.array([1, 2, 1])[: len(right_nodes)] * (b / 2)
    )  # Anpassen an ny
    if len(loads) < len(right_nodes):
        loads = np.tile(loads, (len(right_nodes) // len(loads) + 1))[: len(right_nodes)]
    neum = np.column_stack((right_nodes, np.ones(len(right_nodes)), loads))

    return drlt, neum


def plot_beam(
    coords,
    elems,
    drlt,
    neum,
    l=None,
    b=None,
    nx=None,
    ny=None,
    figsize=(10, 4),
):
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


# Beispiel-Aufruf: Langer, schmaler Balken (sollte jetzt proportional skaliert plotten)
# coords, elems = create_geometry(l=2, b=0.5, h=20, nx=5, ny=5)
# drlt, neum = create_boundary_conditions(l=2, b=0.5, nx=5, ny=5, force=10000)

# plot_beam(
#     coords, elems, drlt, neum, l=2, b=0.5, nx=5, ny=5, figsize=(12, 1)
# )  # figsize angepasst für schmalen Plot

# print(coords, elems, drlt, neum)
