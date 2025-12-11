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
from src.plotting import plot_results, animate_eigenmode



if __name__ == "__main__":
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
    t_start = timemodule.perf_counter()
    solver.run()
    t_end = timemodule.perf_counter()
    print(f"Static Analysis Time: {t_end - t_start:.4f}s")

    # Run Modal Analysis
    print("Run Modal Analysis...")
    t_start = timemodule.perf_counter()
    frequencies, eigenvectors = solver.modal_analysis(num_modes=ANZAHL_MODEN, f_threshold=F_THRESHOLD)
    t_end = timemodule.perf_counter()
    print(f"Modal Analysis Time: {t_end - t_start:.4f}s")
    
    # Tabelle ausgeben (optional, da plot_eigenmodes auch plottet)
    print(f"{'='*40}")
    print(f"{'EIGENFREQUENZEN (Hz)':^40}")
    print(f"{'='*40}")
    print(f"{'Mode':<10} | {'Frequenz (Hz)':<20}")
    print("-" * 40)
    for i, freq in enumerate(frequencies):
        print(f"{i+1:<10} | {freq:.4f}")
    print("-" * 40)

    if TOPLOT:
        from src.plotting import animate_eigenmode
        from config import PLOT_MODE_INDEX
        animate_eigenmode(solver, frequencies, eigenvectors, mode_index=PLOT_MODE_INDEX)

    # Run Transient Analysis (Optional)
    if RUN_TRANSIENT:
        print("Run Transient Analysis...")
        # Erstelle Generator
        gen = solver.newmark_time_integration(
            beta=NEWMARK_BETA, 
            gamma=NEWMARK_GAMMA, 
            dt=TIME_INTERVAL/TIME_STEPS, 
            steps=TIME_STEPS,
            disp_scaling=DISP_SCALING
        )
        
        t_start = timemodule.perf_counter()
        if TOPLOT:
            from src.plotting import animate_time_integration
            # Animation verbraucht Zeit, die nicht zur reinen Berechnung gehört,
            # aber hier ist Berechnung und Plotting gekoppelt durch den Generator.
            # Wir messen die Zeit für den gesamten Animations-Loop.
            animate_time_integration(solver, gen, disp_scaling=DISP_SCALING)
        else:
            # Falls kein Plot gewünscht, Generator einfach durchlaufen lassen (reine Rechenzeit)
            for _ in gen:
                pass
        t_end = timemodule.perf_counter()
        print(f"Transient Analysis (incl. Animation if enabled) Time: {t_end - t_start:.4f}s")

    if TOPLOT:
        print("Plotting Results...")
        plot_results(solver, disp_scaling=DISP_SCALING)
        plt.show()
