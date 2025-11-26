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
    USE_Q8,
    RHO,
    NEN,
    NQP,
    ANZAHL_MODEN,
    F_THRESHOLD,
    RUN_TRANSIENT,
    TIME_INTERVAL,
    TIME_STEPS,
    NEWMARK_BETA,
    NEWMARK_GAMMA,
)
from src.geometry import create_geometry, create_boundary_conditions
from src.material import IsotropicMaterial
from src.mesh import Mesh
from src.solver import FEMSolver
import matplotlib.pyplot as plt
from src.plotting import plot_results, plot_modal_analysis, plot_time_integration


if __name__ == "__main__":
    start_time = timemodule.perf_counter()

    # Setup
    print(f"Setup Geometry (Q8={USE_Q8})...")
    coords, elems = create_geometry(LENGTH, HEIGHT, nx=NX, ny=NY, use_q8=USE_Q8)
    
    # Pass TOTAL_FORCE as line load [N/m]
    drlt, neum = create_boundary_conditions(coords, elems, TOTAL_FORCE, WIDTH)
    
    material = IsotropicMaterial(E, NU)
    mesh = Mesh(coords, elems, drlt, neum)
    
    print("Initialize Solver...")
    solver = FEMSolver(mesh, material, nen=NEN, nqp=NQP, rho=RHO)

    # Run Static Analysis
    print("Run Static Analysis...")
    solver.run()

    # Run Modal Analysis
    print("Run Modal Analysis...")
    frequencies = solver.modal_analysis(num_modes=ANZAHL_MODEN, f_threshold=F_THRESHOLD)
    plot_modal_analysis(frequencies)

    # Run Transient Analysis (Optional)
    if RUN_TRANSIENT:
        print("Run Transient Analysis...")
        time_hist, disp_hist = solver.newmark_time_integration(
            beta=NEWMARK_BETA, 
            gamma=NEWMARK_GAMMA, 
            dt=TIME_INTERVAL/TIME_STEPS, 
            steps=TIME_STEPS,
            disp_scaling=DISP_SCALING
        )
        if TOPLOT:
            plot_time_integration(time_hist, disp_hist)

    if TOPLOT:
        print("Plotting Results...")
        plot_results(solver, disp_scaling=DISP_SCALING)
        plt.show()

    end_time = timemodule.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.4f}s")
