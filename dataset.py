import torch
from torch.utils.data import Dataset
from kornia.utils.grid import create_meshgrid, create_meshgrid3d
from einops import repeat
from patterns import einops_f
import numpy as np


class CoordinateDataset(Dataset):
    def __init__(self, out, hparams, active_blocks=None):
        """
        out: output resized to the current level
             subtracted by the upsampled reconstruction of the previous level
             in finer levels
        active_blocks: torch.tensor, None to return all blocks,
                       otherwise specify the blocks to take
        """
        self.size = np.prod(hparams.patch_size)

        # split into patches
        out = einops_f(out, hparams.patterns['reshape'][3], hparams)
        self.out = torch.tensor(out) # (n, p, c)

        if hparams.task == 'image':
            inp = create_meshgrid(hparams.p2, hparams.p1)
        elif hparams.task == 'mesh':
            inp = create_meshgrid3d(hparams.p3, hparams.p2, hparams.p1)
        self.inp = einops_f(inp, hparams.patterns['reshape'][7])

        if active_blocks is not None:
            self.inp = repeat(self.inp, '1 p c -> n p c', n=len(self.out))
            self.inp = self.inp[active_blocks]
            self.out = self.out[active_blocks]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # each batch contains all blocks with randomly selected cells
        # the cells in each block are at the same position
        return {"inp": self.inp[:, idx], "out": self.out[:, idx]}