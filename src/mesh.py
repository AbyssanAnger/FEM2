import torch
import numpy as np


class Mesh:
    """Handhabt Knoten, Elemente und Randbedingungen."""

    def __init__(self, x_coords, elems):
        self.x = torch.from_numpy(x_coords).float()
        self.elems = torch.from_numpy(elems).long()
        self.nnp = self.x.size(0)
        self.nel = self.elems.size(0)
        print(f"nnp: {self.nnp}, nel: {self.nel}")

    def setup_bcs(self, drlt, neum, ndf=2):
        """Dirichlet (drlt) und Neumann (neum) BCs einrichten."""
        drlt_mask = torch.zeros(self.nnp * ndf, 1)
        drlt_vals = torch.zeros(self.nnp * ndf, 1)
        for i in range(drlt.shape[0]):
            idx = (drlt[i, 0] - 1) * ndf + drlt[i, 1]
            drlt_mask[idx, 0] = 1
            drlt_vals[idx, 0] = drlt[i, 2]
        self.free_mask = 1 - drlt_mask
        self.drlt_mask = drlt_mask
        self.drlt_vals = drlt_vals
        self.drlt_matrix = 1e22 * torch.diag(drlt_mask.squeeze())

        self.neum_vals = torch.zeros(self.nnp * ndf, 1)
        for i in range(neum.shape[0]):
            idx = (neum[i, 0] - 1) * ndf + neum[i, 1]
            self.neum_vals[idx, 0] = neum[i, 2]
