import torch

# Threading and dtype
DEFAULT_DTYPE = torch.float64
TORCH_THREADS = 4
torch.set_default_dtype(DEFAULT_DTYPE)
torch.set_num_threads(TORCH_THREADS)

# Problem parameters
E = 210e9
NU = 0.3
TDM = 2
NDF = 2
NDM = 2
NEN = 4
NQP = 4

# Geometry # in meters
LENGTH = 2.0
HEIGHT = 0.05
WIDTH = 0.05
NX = 400
NY = 10

# Loads / BCs
# total line load in N per meter width applied on right edge, positive downward
FORCE = -1000.0
TOTAL_FORCE = FORCE / WIDTH  # N/m on the boundary (consistent with original)

# Plotting
TOPLOT = True
DISP_SCALING = 50
