import numpy as np


def create_geometry(length=2.0, height=0.05, nx=120, ny=10):
    """Generiert Geometrie (angepasst an Code-Parameter)."""
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    x_coords = np.linspace(0, length, n_nodes_x)
    y_coords = np.linspace(0, height, n_nodes_y)
    X, Y = np.meshgrid(x_coords, y_coords)
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
