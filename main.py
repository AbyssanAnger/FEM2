import time as timemodule
from src.solver import FEMSolver
from src.mesh import Mesh
from src.material import IsotropicMaterial
from config import E, NU, l, b, h, nx, ny, force, y_size, x_size
from src.geometry import create_geometry, create_boundary_conditions, plot_beam
from src.utils import plot_beam, plot_results

import numpy as np

# Input-Daten (wie zuvor)
# x_coords = np.array(
#     [[0.0, 0.0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]
# )
# elems = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
# drlt = np.array([[1, 0, 0], [1, 1, 0], [4, 0, 0], [4, 1, 0], [7, 0, 0], [7, 1, 0]])
# neum = np.array([[3, 1, 5000000], [6, 1, 10000000], [9, 1, 5000000]])

l = 2  # m

if __name__ == "__main__":
    start = timemodule.perf_counter()
    coords, elems = create_geometry(l, b, h, nx, ny)
    drlt, neum = create_boundary_conditions(l, b, nx, ny, force)
    mesh = Mesh(coords, elems)  # Initialisiert Mesh → Print: nnp: 9, nel: 4
    mesh.setup_bcs(drlt, neum)  # Richtet BCs ein (kein Print, aber Attribute gesetzt)
    material = IsotropicMaterial(E, NU)
    solver = FEMSolver(mesh, material)
    solver.run(drlt, neum)  # solver
    figure_1 = plot_beam(coords, elems, drlt, neum, l, b, nx, ny)  # pre-processing

    end = timemodule.perf_counter()
    print(f"Elapsed time: {end - start:.2f}s")
