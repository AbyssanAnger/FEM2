import time as timemodule
from config import (
    TOPLOT,
    E,
    NU,
    LENGTH,
    HEIGHT,
    WIDTH,
    NX,
    NY,
    TOTAL_FORCE,
    DISP_SCALING,
)
from src.geometry import create_geometry, create_boundary_conditions
from src.material import IsotropicMaterial
from src.mesh import Mesh
from src.solver import FEMSolver
import matplotlib.pyplot as plt
from src.utils import plot_results


if __name__ == "__main__":
    start_time = timemodule.perf_counter()

    # Setup
    coords, elems = create_geometry(LENGTH, HEIGHT, nx=NX, ny=NY)
    # Pass TOTAL_FORCE as line load [N/m]
    drlt, neum = create_boundary_conditions(coords, elems, TOTAL_FORCE, WIDTH)
    material = IsotropicMaterial(E, NU)
    mesh = Mesh(coords, elems, drlt, neum)
    solver = FEMSolver(mesh, material)

    # Run
    solver.run()

    if TOPLOT:
        plot_results(solver, disp_scaling=DISP_SCALING)
        plt.show()

    end_time = timemodule.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.4f}s")
