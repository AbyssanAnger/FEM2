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

    if RUN_MODAL:
        print("\n--- Modalanalyse ---")
        omega, modes = solver.compute_modes(approach="M_inv_K", num_modes=NUM_MODES)
        # Optional: Plot erste Mode
        if TOPLOT:
            plot_mode(
                solver, modes[:, 0], mode_num=1, disp_scaling=DISP_SCALING * 10
            )  # Skaliere für Sichtbarkeit
            plt.show()

    # NEU: Transient-Analyse (startet von statischem u, freie Schwingung)
    if RUN_TRANSIENT:
        print("\n--- Transient-Analyse ---")

        def zero_load(t):  # Beispiel: Keine externe Last (freie Schwingung)
            return torch.zeros(solver.ndf * solver.nnp, 1)

        # Oder dynamische Last: def sinus_load(t): return amp * torch.sin(2*math.pi*f*t * fsur)
        history = solver.solve_transient(
            dt=DT,
            total_time=TOTAL_TIME,
            f_ext_func=zero_load,  # Oder sinus_load
            beta=0.25,
            gamma=0.5,
        )
        # Plot History (z. B. finale u oder Animation)
        if TOPLOT:
            plot_transient_history(solver, history, disp_scaling=DISP_SCALING)
            plt.show()

    if TOPLOT:
        plot_results(solver, disp_scaling=DISP_SCALING)
        plt.show()

    end_time = timemodule.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.4f}s")
