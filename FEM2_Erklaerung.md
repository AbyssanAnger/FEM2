# Schritt-für-Schritt Erklärung: FEM2_HA1_Diskretisierer.py

## Übersicht: Wie funktioniert die Finite-Elemente-Methode (FEM)?

Die FEM löst partielle Differentialgleichungen (z.B. Elastizitätsgleichungen) numerisch durch:
1. **Diskretisierung**: Unterteilen des Kontinuums in finite Elemente
2. **Ansatzfunktionen**: Approximation des Verschiebungsfeldes durch Formfunktionen
3. **Elementmatrizen**: Aufstellen der Steifigkeitsmatrix für jedes Element
4. **Assemblierung**: Zusammenfügen aller Elemente zur globalen Steifigkeitsmatrix
5. **Lösung**: Lösen des linearen Gleichungssystems K·u = f
6. **Post-Processing**: Berechnung von Spannungen, Dehnungen etc.

---

## ABSCHNITT 1: Initialisierung und Materialparameter (Zeilen 1-27)

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import time as timemodule

torch.set_default_dtype(torch.float64)  # Doppelte Genauigkeit
torch.set_num_threads(8)                 # Parallelisierung

disp_scaling = 50  # Skalierung für Visualisierung der Verformung
toplot = True
tdm = 2  # Tensor-Dimension (2D)
```

### Materialparameter (Zeilen 17-27)
```python
E = 210e9      # Elastizitätsmodul [Pa] = 210 GPa (Stahl)
NU = 0.3       # Poisson-Zahl (Querkontraktionszahl)
FORCE = -1000.0  # Querkraft [N]
LENGTH = 2.0     # Balkenlänge [m]
HEIGHT = 0.05    # Balkenhöhe [m]
WIDTH = 0.05     # Balkenbreite [m]
NX = 120         # Elemente in x-Richtung
NY = 10          # Elemente in y-Richtung
```

**Bedeutung**: 
- **E**: Beschreibt die Steifigkeit des Materials (Verhältnis Spannung zu Dehnung)
- **NU**: Beschreibt die Querkontraktion (z.B. bei Zug in x-Richtung kommt Kontraktion in y-Richtung)
- **NX, NY**: Feinheit des Netzes (mehr Elemente = genauere Lösung, aber aufwendiger)

---

## ABSCHNITT 2: Mesh-Generierung (Zeilen 32-47)

### 2.1 Knoten generieren (Zeilen 32-36)
```python
x_coords = torch.linspace(0, LENGTH, NX + 1)  # x-Koordinaten: 0 bis 2.0 m
y_coords = torch.linspace(0, HEIGHT, NY + 1)  # y-Koordinaten: 0 bis 0.05 m
x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
x = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1)
```

**Was passiert hier?**
- Erzeugt ein regelmäßiges Gitter von Knoten
- `x_coords`: 121 Punkte von 0 bis 2.0 m (121 = NX+1)
- `y_coords`: 11 Punkte von 0 bis 0.05 m (11 = NY+1)
- `meshgrid`: Erzeugt alle Kombinationen → 121 × 11 = 1331 Knoten
- `x`: Array mit allen Knotenkoordinaten [1331 × 2]

**Visualisierung**:
```
y ↑
  │  ●───●───●───●  (11 Knoten in y-Richtung)
  │  │   │   │   │
  │  ●───●───●───●
  │  │   │   │   │
  │  ●───●───●───●
  └──────────────────→ x
      (121 Knoten in x-Richtung)
```

### 2.2 Elemente generieren (Zeilen 38-47)
```python
elements_list = []
for j in range(NY):
    for i in range(NX):
        n1 = i * (NY + 1) + j      # Unten links
        n2 = (i + 1) * (NY + 1) + j  # Unten rechts
        n3 = (i + 1) * (NY + 1) + j + 1  # Oben rechts
        n4 = i * (NY + 1) + j + 1  # Oben links
        elements_list.append([n1, n2, n3, n4])
elems = torch.tensor(elements_list, dtype=torch.long)
```

**Was passiert hier?**
- Erzeugt 4-Knoten-Rechteckelemente (bilineare Elemente)
- Jedes Element verbindet 4 Knoten
- Nummerierung im Uhrzeigersinn: unten links → unten rechts → oben rechts → oben links
- Insgesamt: NX × NY = 120 × 10 = 1200 Elemente

**Element-Struktur**:
```
n4 ────── n3
│          │
│   El     │
│          │
n1 ────── n2
```

---

## ABSCHNITT 3: Randbedingungen (Zeilen 49-59)

### 3.1 Dirichlet-Randbedingungen (Einspannung) (Zeilen 49-52)
```python
left_nodes = torch.where(x[:, 0] == 0)[0]  # Alle Knoten mit x=0
drlt_list = [[[node_idx, 0, 0], [node_idx, 1, 0]] for node_idx in left_nodes]
drlt = torch.tensor([item for sublist in drlt_list for item in sublist])
```

**Was passiert hier?**
- `drlt` = Dirichlet-Randbedingungen (vorgegebene Verschiebungen)
- Format: `[Knotenindex, Richtung (0=x, 1=y), Verschiebungswert]`
- Alle Knoten am linken Rand (x=0) werden in x- UND y-Richtung eingespannt (Verschiebung = 0)
- Beispiel: `[5, 0, 0]` bedeutet: Knoten 5, x-Richtung, Verschiebung = 0

### 3.2 Neumann-Randbedingungen (Kräfte) (Zeilen 54-59)
```python
total_force = FORCE / WIDTH  # Kraft pro Meter Breite [N/m]
right_nodes = torch.where(x[:, 0] == LENGTH)[0]  # Alle Knoten mit x=LENGTH
force_per_node = total_force / len(right_nodes)
neum_list = [[node_idx, 1, force_per_node] for node_idx in right_nodes]
neum = torch.tensor(neum_list)
```

**Was passiert hier?**
- `neum` = Neumann-Randbedingungen (vorgegebene Kräfte)
- Format: `[Knotenindex, Richtung (0=x, 1=y), Kraftwert]`
- Am rechten Rand (x=LENGTH) wird eine Querkraft in y-Richtung aufgebracht
- Die Gesamtkraft wird gleichmäßig auf alle Knoten am rechten Rand verteilt
- `FORCE = -1000.0 N` → nach unten gerichtet

**Visualisierung**:
```
y ↑
  │  🔒───────→  (eingespannt)
  │  🔒         (Kraft nach unten)
  │  🔒
  └──────────────────→ x
  x=0            x=LENGTH
```

---

## ABSCHNITT 4: Materialtensor (Zeilen 64-73)

```python
C4 = torch.zeros(2, 2, 2, 2)  # 4. Stufe Tensor
factor = E / (1 - NU**2)
C4[0, 0, 0, 0] = factor * 1      # σ_xx = C_xxxx * ε_xx
C4[1, 1, 1, 1] = factor * 1      # σ_yy = C_yyyy * ε_yy
C4[0, 0, 1, 1] = factor * NU     # σ_xx = C_xxyy * ε_yy
C4[1, 1, 0, 0] = factor * NU     # σ_yy = C_yyxx * ε_xx

shear_factor = E / (2 * (1 + NU))
C4[0, 1, 0, 1] = ... = shear_factor  # Schubspannung
```

**Was ist das?**
- **C4**: Elastizitätstensor 4. Stufe (Hookesches Gesetz: σ = C : ε)
- **Ebener Spannungszustand** (plane stress): σ_zz = 0
- Verknüpft Spannungen mit Dehnungen: σ_ij = C_ijkl · ε_kl
- `factor = E/(1-ν²)`: Materialkonstante für ebene Spannung

**Mathematisch**:
```
σ_xx = E/(1-ν²) · (ε_xx + ν·ε_yy)
σ_yy = E/(1-ν²) · (ε_yy + ν·ε_xx)
σ_xy = E/(2(1+ν)) · ε_xy
```

---

## ABSCHNITT 5: DOF-Verwaltung (Zeilen 79-93)

### 5.1 Dirichlet-Masken (Zeilen 79-89)
```python
drlt_mask = torch.zeros(nnp * NDF, 1)  # Maske: 1 = eingespannt, 0 = frei
drlt_vals = torch.zeros(nnp * NDF, 1)  # Vorgegebene Verschiebungswerte

for i in range(drlt.size()[0]):
    drlt_mask[int(drlt[i, 0]) * NDF + int(drlt[i, 1]), 0] = 1
    drlt_vals[int(drlt[i, 0]) * NDF + int(drlt[i, 1]), 0] = drlt[i, 2]

free_mask = torch.ones(nnp * NDF, 1) - drlt_mask  # Inverse Maske
drlt_matrix = 1e22 * torch.diag(drlt_mask[:, 0], 0)  # Penalty-Matrix
```

**Was passiert hier?**
- **DOF** = Degrees of Freedom (Freiheitsgrade)
- Jeder Knoten hat 2 DOF: u_x und u_y
- Gesamtanzahl DOF: `nnp * NDF = 1331 * 2 = 2662`
- `drlt_mask`: Markiert welche DOF eingespannt sind (1) oder frei (0)
- `drlt_matrix`: Sehr große Zahlen auf der Diagonalen für eingespannte DOF (Penalty-Methode)

**Beispiel**:
```
Knoten 0: DOF 0 (u_x), DOF 1 (u_y)
Knoten 1: DOF 2 (u_x), DOF 3 (u_y)
...
Knoten i: DOF 2*i (u_x), DOF 2*i+1 (u_y)
```

### 5.2 Neumann-Kräfte (Zeilen 91-93)
```python
neum_vals = torch.zeros(nnp * NDF, 1)
for i in range(neum.size()[0]):
    neum_vals[int(neum[i, 0]) * NDF + int(neum[i, 1]), 0] = neum[i, 2]
```

**Was passiert hier?**
- Speichert alle äußeren Kräfte in einem Vektor
- Format: `neum_vals[DOF_index] = Kraftwert`

---

## ABSCHNITT 6: Gauß-Quadratur (Zeilen 95-126)

### 6.1 Gauß-Punkte (Zeilen 95-104)
```python
qpt = torch.zeros(nqp, NDM)  # nqp = 4 Gauß-Punkte
a = math.sqrt(3) / 3  # ≈ 0.577
qpt[0, 0] = -a; qpt[0, 1] = -a  # GP1: (-a, -a)
qpt[1, 0] = a;  qpt[1, 1] = -a  # GP2: (a, -a)
qpt[2, 0] = -a; qpt[2, 1] = a   # GP3: (-a, a)
qpt[3, 0] = a;  qpt[3, 1] = a   # GP4: (a, a)
w8 = torch.ones(nqp, 1)  # Gewichte (alle = 1.0 für 2×2 Quadratur)
```

**Was passiert hier?**
- **2×2 Gauß-Quadratur**: 4 Integrationspunkte im Masterelement
- Koordinaten im Masterelement (ξ, η ∈ [-1, 1])
- Gewichte: alle = 1.0 (für 2×2 Quadratur)

**Visualisierung** (Masterelement):
```
η ↑
  │
 1│    GP3 ●───────● GP4
  │         │       │
  │         │   ╱   │
  │         │  ╱    │
 0│    GP1 ●───────● GP2
  │
-1└────────────────────────→ ξ
 -1         0         1
```

### 6.2 Formfunktionen (Shape Functions) (Zeilen 108-126)
```python
masterelem_N = torch.zeros(nqp, 4)      # Formfunktionen
masterelem_gamma = torch.zeros(nqp, 4, 2)  # Ableitungen der Formfunktionen

for q in range(nqp):
    xi = qpt[q, :]
    # Formfunktionen (bilinear):
    masterelem_N[q, 0] = 0.25 * (1 - xi[0]) * (1 - xi[1])  # N1
    masterelem_N[q, 1] = 0.25 * (1 + xi[0]) * (1 - xi[1])  # N2
    masterelem_N[q, 2] = 0.25 * (1 + xi[0]) * (1 + xi[1])  # N3
    masterelem_N[q, 3] = 0.25 * (1 - xi[0]) * (1 + xi[1])  # N4
    
    # Ableitungen (für B-Matrix):
    masterelem_gamma[q, 0, 0] = -0.25 * (1 - xi[1])  # ∂N1/∂ξ
    masterelem_gamma[q, 0, 1] = -0.25 * (1 - xi[0])  # ∂N1/∂η
    # ... etc.
```

**Was sind Formfunktionen?**
- **Formfunktionen N_i(ξ,η)**: Beschreiben die Verschiebung im Element
- **Eigenschaften**:
  - N_i(ξ_j, η_j) = 1 am Knoten i, 0 an allen anderen Knoten
  - Σ N_i = 1 (Partition of Unity)
- **Verschiebungsansatz**: u(ξ,η) = Σ N_i(ξ,η) · u_i
- **Ableitungen γ**: Werden für die B-Matrix benötigt (Dehnungs-Verschiebungs-Beziehung)

**Mathematisch**:
```
u_x(ξ,η) = N1(ξ,η)·u_x1 + N2(ξ,η)·u_x2 + N3(ξ,η)·u_x3 + N4(ξ,η)·u_x4
u_y(ξ,η) = N1(ξ,η)·u_y1 + N2(ξ,η)·u_y2 + N3(ξ,η)·u_y3 + N4(ξ,η)·u_y4
```

---

## ABSCHNITT 7: DOF-Mapping (Zeilen 133-139)

```python
edof = torch.zeros(nel, NDF, NEN, dtype=int)  # Element DOF
gdof = torch.zeros(nel, NDF * NEN, dtype=int)  # Global DOF

for el in range(nel):
    for ien in range(NEN):
        for idf in range(NDF):
            edof[el, idf, ien] = NDF * elems[el, ien] + idf
    gdof[el, :] = edof[el, :, :].t().reshape(NDF * NEN)
```

**Was passiert hier?**
- **edof**: Verknüpft lokale Element-DOF mit globalen DOF
- **gdof**: Liste aller globalen DOF eines Elements (für Assemblierung)
- Jedes Element hat 4 Knoten × 2 DOF = 8 DOF

**Beispiel**:
```
Element 0 mit Knoten [10, 11, 22, 21]:
  gdof[0] = [20, 21, 22, 23, 44, 45, 42, 43]
           (u_x10, u_y10, u_x11, u_y11, u_x22, u_y22, u_x21, u_y21)
```

---

## ABSCHNITT 8: Element-Schleife - Kern der FEM (Zeilen 156-204)

### 8.1 Initialisierung (Zeilen 156-164)
```python
for el in range(nel):
    xe = torch.zeros(NDM, NEN)  # Knotenkoordinaten des Elements
    for idm in range(NDM):
        xe[idm, :] = x[elems[el, :], idm]
    
    ue = torch.squeeze(u[edof[el, 0:NDM, :]])  # Verschiebungen des Elements
    
    Ke.zero_()  # Element-Steifigkeitsmatrix zurücksetzen
    finte.zero_()  # Element-Innere-Kräfte zurücksetzen
```

**Was passiert hier?**
- Für jedes Element werden die Knotenkoordinaten extrahiert
- `xe`: Koordinaten der 4 Knoten des Elements [2 × 4]
- `ue`: Verschiebungen der 4 Knoten [2 × 4]

### 8.2 Gauß-Punkt-Schleife (Zeilen 165-199)

#### 8.2.1 Jacobi-Matrix und Transformation (Zeilen 166-177)
```python
for q in range(nqp):
    gamma = masterelem_gamma[q]  # Ableitungen im Masterelement
    
    Je = xe.mm(gamma)  # Jacobi-Matrix: J = [∂x/∂ξ  ∂x/∂η]
                       #                  [∂y/∂ξ  ∂y/∂η]
    detJe = torch.det(Je)  # Determinante (für Volumen-Transformation)
    
    dv = detJe * w8[q]  # Volumenelement: dV = det(J) · dξdη · Gewicht
    invJe = torch.inverse(Je)  # Inverse Jacobi-Matrix
    
    G = gamma.mm(invJe)  # B-Matrix Komponenten: G = [∂N/∂x]
                         #                            [∂N/∂y]
```

**Was ist die Jacobi-Matrix?**
- **Transformation** vom Masterelement (ξ,η) zum realen Element (x,y)
- **Je**: Beschreibt die Verzerrung des Elements
- **detJe**: Maß für die Größe des Elements (wird für Volumen-Integration benötigt)
- **G**: Gradienten-Matrix (Ableitungen nach x,y statt ξ,η)

**Mathematisch**:
```
x(ξ,η) = Σ N_i(ξ,η) · x_i
J = [∂x/∂ξ  ∂x/∂η] = [Σ ∂N_i/∂ξ · x_i  Σ ∂N_i/∂η · x_i]
    [∂y/∂ξ  ∂y/∂η]   [Σ ∂N_i/∂ξ · y_i  Σ ∂N_i/∂η · y_i]

∂N/∂x = (∂N/∂ξ) · (∂ξ/∂x) = γ · J^(-1)
```

#### 8.2.2 Dehnungen und Spannungen (Zeilen 179-181)
```python
h[0:NDM, 0:NDM] = ue.mm(G)  # Verschiebungsgradient: h = ∇u
eps_2d = 0.5 * (h + h.t())  # Dehnungstensor: ε = 0.5·(∇u + ∇u^T)
stre_2d = torch.tensordot(C4, eps_2d, dims=2)  # Spannung: σ = C : ε
```

**Was passiert hier?**
- **h = ∇u**: Verschiebungsgradient (nicht symmetrisch)
- **ε = 0.5·(∇u + ∇u^T)**: Linearisierter Dehnungstensor (symmetrisch)
- **σ = C : ε**: Hookesches Gesetz (Spannung aus Dehnung)

**Komponenten**:
```
ε_xx = ∂u_x/∂x
ε_yy = ∂u_y/∂y
ε_xy = 0.5·(∂u_x/∂y + ∂u_y/∂x)

σ_xx = C_xxxx·ε_xx + C_xxyy·ε_yy
σ_yy = C_yyxx·ε_xx + C_yyyy·ε_yy
σ_xy = C_xyxy·ε_xy
```

#### 8.2.3 Innere Kräfte (Zeilen 183-187)
```python
for A in range(NEN):
    G_A[0, :] = G[A, :]  # Gradient der Formfunktion A
    
    fintA = dv * (G_A.mm(stre_2d))  # Beitrag zur inneren Kraft
    finte[A * NDF : A * NDF + NDM] += fintA.t()
```

**Was passiert hier?**
- **Innere Kräfte**: f_int = ∫ B^T · σ dV
- Für jeden Knoten A wird der Beitrag zur inneren Kraft berechnet
- Integration über das Element mittels Gauß-Quadratur

**Mathematisch**:
```
f_int^A = ∫ [∂N_A/∂x  ∂N_A/∂y] · [σ_xx  σ_xy]^T dV
                        [σ_xy  σ_yy]
```

#### 8.2.4 Steifigkeitsmatrix (Zeilen 189-199)
```python
for B in range(NEN):
    KAB = torch.tensordot(
        G[A, :],
        (torch.tensordot(C4[0:tdm, 0:tdm, 0:tdm, 0:tdm], G[B, :], [[3], [0]])),
        [[0], [0]],
    )
    Ke[A * NDF : A * NDF + NDM, B * NDF : B * NDF + NDM] += dv * KAB
```

**Was passiert hier?**
- **Steifigkeitsmatrix**: K^e = ∫ B^T · C · B dV
- **KAB**: Beitrag von Knoten A zu Knoten B
- Integration über das Element mittels Gauß-Quadratur

**Mathematisch**:
```
K_AB = ∫ [∂N_A/∂x  ∂N_A/∂y] · C · [∂N_B/∂x] dV
                           [∂N_B/∂y]
```

### 8.3 Assemblierung (Zeilen 201-204)
```python
fint[gdof[el, :]] += finte  # Innere Kräfte zum globalen Vektor
fvol[gdof[el, :]] += fvole  # Volumenkräfte (hier = 0)
for i in range(gdof.shape[1]):
    K[gdof[el, i], gdof[el, :]] += Ke[i, :]  # Steifigkeitsmatrix assembliert
```

**Was passiert hier?**
- **Assemblierung**: Zusammenfügen aller Elemente zur globalen Matrix
- Element-Steifigkeitsmatrizen werden in die globale Steifigkeitsmatrix eingefügt
- Element-Kräfte werden zum globalen Kraftvektor addiert

**Visualisierung**:
```
Global K = [K^1] + [K^2] + [K^3] + ... + [K^nel]
           ↓       ↓       ↓              ↓
         Element  Element  Element      Element
           1        2        3          nel
```

---

## ABSCHNITT 9: Lösung des Gleichungssystems (Zeilen 208-214)

```python
rsd = free_mask.mul(fsur - fint)  # Residuum: R = f_ext - f_int

K_tilde = K + drlt_matrix  # Steifigkeitsmatrix mit Penalty-Termen

du = torch.linalg.solve(K_tilde, rsd)  # Lösen: K · Δu = R
u += du  # Verschiebung aktualisieren
u = free_mask.mul(u) + drlt_mask.mul(drlt_vals)  # Randbedingungen setzen
```

**Was passiert hier?**
- **Residuum R**: Differenz zwischen äußeren und inneren Kräften
- **Penalty-Methode**: Sehr große Zahlen auf der Diagonalen für eingespannte DOF
- **Lösung**: K · u = f wird gelöst
- **Randbedingungen**: Eingespannte DOF werden auf vorgegebene Werte gesetzt

**Mathematisch**:
```
K · u = f_ext

Mit Randbedingungen:
[K + P] · u = f_ext + P · u_0

wobei P = diag(1e22) für eingespannte DOF
```

**Warum Penalty-Methode?**
- Einfacher als Elimination von DOF
- Die großen Zahlen zwingen u ≈ u_0 an eingespannten Stellen

---

## ABSCHNITT 10: Post-Processing - Spannungen (Zeilen 230-304)

### 10.1 Extrapolation von Gauss-Punkten zu Knoten (Zeilen 230-236)
```python
nodal_stresses_sum = torch.zeros(nnp, len(plot_titles))
nodal_contribution_count = torch.zeros(nnp, len(plot_titles))
N_at_gps = torch.vstack([masterelem_N[q] for q in range(nqp)])
extrapolation_matrix = torch.inverse(N_at_gps)  # Extrapolation: GP → Knoten
```

**Was passiert hier?**
- Spannungen sind an Gauss-Punkten bekannt (genauer)
- Für Visualisierung werden Knotenwerte benötigt
- **Extrapolation**: Umrechnung von GP-Werten zu Knotenwerten
- `extrapolation_matrix`: Invertiert die Formfunktionen an den GP-Positionen

### 10.2 Spannungen an Gauss-Punkten berechnen (Zeilen 239-280)
```python
for e in range(nel):
    # ... Koordinaten und Verschiebungen extrahieren ...
    
    for q in range(nqp):
        # ... Jacobi-Matrix, Dehnungen, Spannungen berechnen ...
        
        s11 = stre_2d[0, 0]  # σ_xx
        s22 = stre_2d[1, 1]  # σ_yy
        s12 = stre_2d[0, 1]  # σ_xy
        s33 = 0.0  # σ_zz = 0 (ebener Spannungszustand)
        
        # Von-Mises-Spannung:
        sigma_v = torch.sqrt(
            0.5 * (
                (s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 + 6*(s12**2)
            )
        )
        
        s_xx_gp[q] = s11
        s_yy_gp[q] = s22
        s_vm_gp[q] = sigma_v
        eps_xx_gp[q] = eps_2d[0, 0]
```

**Was passiert hier?**
- Für jedes Element werden Spannungen an allen 4 Gauss-Punkten berechnet
- **Von-Mises-Spannung**: Vergleichsspannung für Festigkeitsnachweis
- `sigma_v = sqrt(0.5·[(σ_xx-σ_yy)² + (σ_yy-σ_zz)² + (σ_zz-σ_xx)² + 6·σ_xy²])`

### 10.3 Extrapolation und Mittelung (Zeilen 282-304)
```python
nodal_vals_elem = torch.stack([
    extrapolation_matrix.mv(s_xx_gp),  # GP → Knoten
    extrapolation_matrix.mv(s_yy_gp),
    extrapolation_matrix.mv(s_vm_gp),
    extrapolation_matrix.mv(eps_xx_gp),
])

element_nodes = elems[e, :]
nodal_stresses_sum[element_nodes, 0] += nodal_vals_elem[0]  # Aufsummieren
nodal_stresses_sum[element_nodes, 1] += nodal_vals_elem[1]
# ... etc.

# Mittelung (wenn Knoten zu mehreren Elementen gehört):
counts = nodal_contribution_count.clone()
counts[counts == 0] = 1
nodal_avg = nodal_stresses_sum / counts
```

**Was passiert hier?**
- **Extrapolation**: GP-Werte → Knotenwerte pro Element
- **Mittelung**: Knoten, die zu mehreren Elementen gehören, erhalten gemittelte Werte
- Ergebnis: Glatte Spannungsfelder an Knoten

**Warum Mittelung?**
- Ein Knoten gehört zu mehreren Elementen
- Jedes Element liefert einen Wert
- Mittelung liefert konsistente Knotenwerte

---

## ABSCHNITT 11: Visualisierung (Zeilen 306-414)

### 11.1 Hilfsfunktionen (Zeilen 307-332)
```python
def plot_mesh(ax, coords, connectivity, idx_seq):
    # Zeichnet das Netz
    
def plot_scalar(ax, values, title, unit=None, scale=None):
    # Zeichnet skalare Felder als Konturplots
```

### 11.2 Plots (Zeilen 334-412)
- **Subplot 1**: Unverformtes Netz mit Randbedingungen
- **Subplot 2-4**: Spannungen (σ_xx, σ_yy, Von-Mises)
- **Subplot 5-6**: Verschiebungen (u_x, u_y)
- **Subplot 7**: Dehnungen (ε_xx)
- **Subplot 8**: Vergleich mit analytischer Lösung (Biegespannung)

**Analytische Lösung** (Zeilen 394-397):
```python
I_z = (WIDTH * HEIGHT**3) / 12  # Flächenträgheitsmoment
M_z = (1000.0) * (LENGTH - mid_point_x)  # Biegemoment
sigma_analytical = (M_z / I_z * (y_analytical - HEIGHT / 2)) / 1e6
```

**Bedeutung**:
- **Balkentheorie**: σ = M·y / I
- Vergleich zwischen FEM und analytischer Lösung
- Validierung der FEM-Implementierung

---

## Zusammenfassung: FEM-Ablauf

1. **Preprocessing**:
   - Mesh generieren
   - Randbedingungen definieren
   - Materialparameter setzen

2. **Element-Loop**:
   - Für jedes Element:
     - Steifigkeitsmatrix K^e berechnen
     - Innere Kräfte f_int^e berechnen
     - In globale Matrix/Vektoren assembliert

3. **Lösung**:
   - K · u = f lösen
   - Randbedingungen berücksichtigen

4. **Post-Processing**:
   - Spannungen/Dehnungen berechnen
   - Visualisieren
   - Mit analytischer Lösung vergleichen

---

## Wichtige Formeln

**Verschiebungsansatz**:
```
u(ξ,η) = Σ N_i(ξ,η) · u_i
```

**Dehnungen**:
```
ε = 0.5 · (∇u + ∇u^T)
```

**Spannungen (Hookesches Gesetz)**:
```
σ = C : ε
```

**Steifigkeitsmatrix**:
```
K^e = ∫ B^T · C · B dV
```

**Innere Kräfte**:
```
f_int = ∫ B^T · σ dV
```

**Gleichgewicht**:
```
K · u = f_ext
```

---

## Warum funktioniert die FEM?

1. **Diskretisierung**: Kontinuum → endlich viele Knoten
2. **Ansatzfunktionen**: Verschiebungsfeld wird approximiert
3. **Variationsformulierung**: Schwache Form der DGL
4. **Galerkin-Methode**: Residuum wird minimiert
5. **Numerische Integration**: Gauß-Quadratur für Integrale

Die FEM ist eine **numerische Methode**, die die **exakte Lösung** einer partiellen Differentialgleichung durch eine **Näherungslösung** ersetzt, die aber bei feinerem Netz immer genauer wird.

