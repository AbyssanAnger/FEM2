import torch
import numpy as np


class Mesh:
    """Handhabt Knoten, Elemente und Randbedingungen."""

    def __init__(self, coords, elems):
        self.x = torch.from_numpy(coords).float()
        self.elems = torch.from_numpy(elems).long()
        self.nnp = self.x.size(0)
        self.nel = self.elems.size(0)
        print(f"nnp: {self.nnp}, nel: {self.nel}")

    def setup_bcs(self, drlt, neum, ndf=2):
        """Dirichlet (drlt) und Neumann (neum) BCs einrichten."""
        total_dofs = self.nnp * ndf
        drlt_mask = torch.zeros(total_dofs, 1)
        drlt_vals = torch.zeros(total_dofs, 1)
        for i in range(drlt.shape[0]):
            idx = int(drlt[i, 0] * ndf + drlt[i, 1])  # 0-basiert, cast zu int
            drlt_mask[idx, 0] = 1
            drlt_vals[idx, 0] = drlt[i, 2]
        self.free_mask = 1 - drlt_mask
        self.drlt_mask = drlt_mask
        self.drlt_vals = drlt_vals
        self.drlt_matrix = 1e22 * torch.diag(drlt_mask.squeeze())  # penalty method

        self.neum_vals = torch.zeros(total_dofs, 1)
        for i in range(neum.shape[0]):
            idx = int(neum[i, 0] * ndf + neum[i, 1])  # 0-basiert, cast zu int
            self.neum_vals[idx, 0] = neum[i, 2]
