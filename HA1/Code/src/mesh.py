import torch
from config import DEFAULT_DTYPE


class Mesh:
    def __init__(self, nodes, elems, drlt=None, neum=None, ndf=2):
        # Convert inputs to torch tensors with appropriate dtypes
        if not isinstance(nodes, torch.Tensor):
            self.x = torch.tensor(nodes, dtype=DEFAULT_DTYPE)
        else:
            self.x = nodes.to(dtype=DEFAULT_DTYPE)

        if not isinstance(elems, torch.Tensor):
            self.elems = torch.tensor(elems, dtype=torch.long)
        else:
            self.elems = elems.to(dtype=torch.long)

        self.ndf = ndf
        self.nnp = self.x.size(0)
        self.nel = self.elems.size(0)

        # Boundary condition definitions (as numpy or torch); store as tensors later
        self._drlt_in = drlt
        self._neum_in = neum

        # Will be initialized in setup_bcs
        self.drlt_mask = None
        self.drlt_vals = None
        self.free_mask = None
        self.drlt_matrix = None
        self.neum_vals = None

    def left_nodes(self):
        return torch.where(self.x[:, 0] == 0)[0]

    def right_nodes(self, length):
        return torch.where(self.x[:, 0] == length)[0]

    def setup_bcs(self):
        """Prepare Dirichlet and Neumann boundary condition vectors/masks."""
        # Convert provided BC arrays to tensors
        if self._drlt_in is None:
            drlt = torch.zeros((0, 3), dtype=DEFAULT_DTYPE)
        else:
            drlt = (
                torch.tensor(self._drlt_in)
                if not isinstance(self._drlt_in, torch.Tensor)
                else self._drlt_in
            )
        if drlt.dtype != torch.long and drlt.numel() > 0:
            # columns 0 and 1 are indices; ensure integer for indexing
            drlt = drlt.to(dtype=DEFAULT_DTYPE)
        # Keep a long/int version for indexing
        drlt_idx = drlt.clone()
        if drlt_idx.numel() > 0:
            drlt_idx[:, 0:2] = drlt_idx[:, 0:2].round()
            drlt_idx = drlt_idx.to(dtype=torch.long)

        if self._neum_in is None:
            neum = torch.zeros((0, 3), dtype=DEFAULT_DTYPE)
        else:
            neum = (
                torch.tensor(self._neum_in)
                if not isinstance(self._neum_in, torch.Tensor)
                else self._neum_in
            )
        if neum.dtype != DEFAULT_DTYPE and neum.numel() > 0:
            neum = neum.to(dtype=DEFAULT_DTYPE)

        # Allocate masks and vectors
        self.drlt_mask = torch.zeros(self.nnp * self.ndf, 1, dtype=DEFAULT_DTYPE)
        self.drlt_vals = torch.zeros(self.nnp * self.ndf, 1, dtype=DEFAULT_DTYPE)

        # Fill Dirichlet mask and values
        if drlt_idx.numel() > 0:
            for i in range(drlt_idx.size(0)):
                node = int(drlt_idx[i, 0].item())
                dof = int(drlt_idx[i, 1].item())
                val = float(drlt[i, 2].item())
                self.drlt_mask[node * self.ndf + dof, 0] = 1.0
                self.drlt_vals[node * self.ndf + dof, 0] = val

        self.free_mask = (
            torch.ones(self.nnp * self.ndf, 1, dtype=DEFAULT_DTYPE) - self.drlt_mask
        )
        # Penalty matrix for constrained dofs
        self.drlt_matrix = 1e22 * torch.diag(self.drlt_mask[:, 0])

        # Neumann values vector (external forces)
        self.neum_vals = torch.zeros(self.nnp * self.ndf, 1, dtype=DEFAULT_DTYPE)
        if neum.numel() > 0:
            neum_idx = neum.clone()
            neum_idx[:, 0:2] = neum_idx[:, 0:2].round()
            neum_idx = neum_idx.to(dtype=torch.long)
            for i in range(neum_idx.size(0)):
                node = int(neum_idx[i, 0].item())
                dof = int(neum_idx[i, 1].item())
                val = float(neum[i, 2].item())
                self.neum_vals[node * self.ndf + dof, 0] += val
