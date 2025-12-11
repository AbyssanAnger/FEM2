import torch
import matplotlib.pyplot as plt
import time as timemodule
from src.geometry import create_geometry, create_boundary_conditions
from src.material import IsotropicMaterial
from src.mesh import Mesh
from src.solver import FEMSolver
from config import (
    E, NU, RHO, LENGTH, HEIGHT, WIDTH, TOTAL_FORCE, 
    USE_Q8, NEN, NQP, ANZAHL_MODEN, F_THRESHOLD
)

def run_convergence_study(start_nx=5, end_nx=50, step_nx=5):
    """
    Führt eine Konvergenzstudie durch, indem die Netzdichte variiert wird.
    
    Args:
        start_nx (int): Startanzahl der Elemente in x-Richtung.
        end_nx (int): Endanzahl der Elemente in x-Richtung.
        step_nx (int): Schrittweite für nx.
    """
    print(f"{'='*60}")
    print(f"{'KONVERGENZSTUDIE (h-Refinement)':^60}")
    print(f"{'='*60}")
    
    results = {
        "dofs": [],
        "u_y_tip": [],
        "f1": [],
        "nx": []
    }
    
    nx_values = range(start_nx, end_nx + 1, step_nx)
    
    for nx in nx_values:
        # Aspect Ratio beibehalten: ny skaliert mit nx
        ny = max(1, int(nx / 10))
        
        print(f"Running simulation for NX={nx}, NY={ny}...")
        
        # 1. Setup
        coords, elems = create_geometry(LENGTH, HEIGHT, nx=nx, ny=ny, use_q8=USE_Q8)
        drlt, neum = create_boundary_conditions(coords, elems, TOTAL_FORCE, WIDTH)
        material = IsotropicMaterial(E, NU)
        mesh = Mesh(coords, elems, drlt, neum)
        solver = FEMSolver(mesh, material, nen=NEN, nqp=NQP, rho=RHO)
        
        # 2. Statische Analyse
        solver.run()
        
        # Verschiebung u_y am unteren rechten Eck (x=L, y=0) oder ähnlich
        # Wir suchen den Knoten bei x=L, y=H/2 (Mitte) oder einfach den Knoten mit max Verschiebung
        # Hier: Knoten am freien Ende (x=L) in der Mitte (y=H/2)
        # Da wir nicht sicher sind, ob ein Knoten genau da liegt, nehmen wir den Knoten am nächsten dran.
        
        # Einfacher: Maximale Verschiebung (Betrag)
        u_mag = torch.sqrt(solver.u[0::2]**2 + solver.u[1::2]**2)
        max_u = torch.max(u_mag).item()
        
        # Oder spezifisch u_y am Ende (wie im Skript oft gesucht)
        # Wir nehmen den letzten Knoten (oben rechts) oder suchen Knoten bei x=L
        right_nodes = torch.where(torch.abs(mesh.x[:, 0] - LENGTH) < 1e-6)[0]
        # Mittelwert der u_y Verschiebung am rechten Rand
        u_y_right = torch.mean(solver.u[right_nodes * 2 + 1]).item()
        
        # 3. Modalanalyse
        frequencies, _ = solver.modal_analysis(num_modes=1, f_threshold=F_THRESHOLD)
        f1 = frequencies[0].item() if len(frequencies) > 0 else 0.0
        
        # Speichern
        dofs = mesh.nnp * 2
        results["dofs"].append(dofs)
        results["u_y_tip"].append(u_y_right)
        results["f1"].append(f1)
        results["nx"].append(nx)
        
        print(f"  -> DOFs: {dofs}, u_y_tip: {u_y_right:.6f} m, f1: {f1:.4f} Hz")
        
    # Plotting
    plot_convergence(results)

def plot_convergence(results):
    """Plottet die Ergebnisse der Konvergenzstudie."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Verschiebung vs DOFs
    ax1.plot(results["dofs"], results["u_y_tip"], 'bo-', label="FEM u_y (Tip)")
    ax1.set_title("Konvergenz der Verschiebung")
    ax1.set_xlabel("Anzahl Freiheitsgrade (DOFs)")
    ax1.set_ylabel("Verschiebung u_y [m]")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Eigenfrequenz vs DOFs
    ax2.plot(results["dofs"], results["f1"], 'rs-', label="FEM f_1")
    ax2.set_title("Konvergenz der 1. Eigenfrequenz")
    ax2.set_xlabel("Anzahl Freiheitsgrade (DOFs)")
    ax2.set_ylabel("Frequenz [Hz]")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_convergence_study()
