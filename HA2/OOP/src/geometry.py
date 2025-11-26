import numpy as np


def create_geometry(length=2.0, height=0.05, nx=120, ny=10, use_q8=False):
    """Generiert Geometrie (angepasst an Code-Parameter)."""
    
    if use_q8:
        # Für Q8-Elemente benötigen wir Knoten an den Ecken UND in der Mitte der Kanten.
        # Wir erzeugen daher ein feineres Gitter von potenziellen Knotenpositionen.
        x_coords = np.linspace(0, length, 2 * nx + 1)
        y_coords = np.linspace(0, height, 2 * ny + 1)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        
        # Knotenkoordinaten als Array [nnp x 2]
        coords_all = np.column_stack((X.ravel(), Y.ravel()))
        
        # Erzeuge Elemente (Konnektivität)
        elems_list = []
        num_nodes_y = 2 * ny + 1
        
        for j in range(ny):  # Schleife über Elemente in y-Richtung
            for i in range(nx):  # Schleife über Elemente in x-Richtung
                # Eckknoten (wie bei Q4, aber mit Schritt 2)
                n1 = (2 * i) * num_nodes_y + (2 * j)        # Unten links
                n2 = (2 * (i + 1)) * num_nodes_y + (2 * j)  # Unten rechts
                n3 = (2 * (i + 1)) * num_nodes_y + (2 * (j + 1)) # Oben rechts
                n4 = (2 * i) * num_nodes_y + (2 * (j + 1))  # Oben links

                # Mittenknoten
                n5 = (2 * i + 1) * num_nodes_y + (2 * j)      # Mitte unten
                n6 = (2 * (i + 1)) * num_nodes_y + (2 * j + 1) # Mitte rechts
                n7 = (2 * i + 1) * num_nodes_y + (2 * (j + 1)) # Mitte oben
                n8 = (2 * i) * num_nodes_y + (2 * j + 1)      # Mitte links

                # Reihenfolge für Q8: 4 Ecken, dann 4 Mittenknoten
                elems_list.append([n1, n2, n3, n4, n5, n6, n7, n8])
        
        elems = np.array(elems_list)
        
        # Filtere ungenutzte Knoten heraus
        unique_nodes = np.unique(elems.flatten())
        coords = coords_all[unique_nodes]

        # Erstelle eine Mapping-Tabelle von alten zu neuen Knotenindizes
        node_map = np.zeros(coords_all.shape[0], dtype=int)
        node_map[unique_nodes] = np.arange(len(unique_nodes))

        # Wende das Mapping auf die Element-Konnektivität an
        elems = node_map[elems]
        
        return coords, elems

    else:
        # Standard Q4 Generierung
        n_nodes_x = nx + 1
        n_nodes_y = ny + 1
        x_coords = np.linspace(0, length, n_nodes_x)
        y_coords = np.linspace(0, height, n_nodes_y)
        X, Y = np.meshgrid(x_coords, y_coords) # Default indexing='xy' -> rows correspond to y
        
        # Achtung: Meshgrid default ist 'xy', d.h. X hat shape (ny, nx). 
        # Um Konsistenz mit Q8 (indexing='ij') oder der Logik unten zu wahren, müssen wir aufpassen.
        # Der ursprüngliche Code nutzte X.ravel(), Y.ravel().
        # Wir behalten die ursprüngliche Logik bei für Q4.
        
        coords = np.column_stack((X.ravel(), Y.ravel()))

        elems = []
        for i in range(ny):
            for j in range(nx):
                n1 = i * n_nodes_x + j
                n2 = i * n_nodes_x + (j + 1)
                n3 = (i + 1) * n_nodes_x + (j + 1)
                n4 = (i + 1) * n_nodes_x + j
                elems.append([n1, n2, n3, n4])
        elems = np.array(elems)

        return coords, elems


def create_boundary_conditions(coords, elems, total_force=-1000.0, width=0.05, ndf=2):
    """Generiert BCs (Fixed left in x/y, Force right in y).

    total_force is interpreted as line load [N/m] on the right edge.
    """
    nnp = coords.shape[0]
    left_nodes = np.where(coords[:, 0] == 0)[0]
    drlt_list = []
    for node in left_nodes:
        drlt_list.extend([[node, 0, 0], [node, 1, 0]])  # Fixed u_x, u_y
    drlt = np.array(drlt_list)

    right_nodes = np.where(coords[:, 0] == coords[:, 0].max())[0]
    # Distribute line load [N/m] equally over boundary nodes
    force_per_node = total_force / len(right_nodes)
    neum_list = [[node, 1, force_per_node] for node in right_nodes]  # y-Force
    neum = np.array(neum_list)

    return drlt, neum
