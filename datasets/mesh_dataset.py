import torch
from torch.utils.data import Dataset
from einops import rearrange
import numpy as np
from typing import Tuple


class MeshDataset(Dataset):
    def __init__(self,
                 occ: np.ndarray,
                 patch_wh: Tuple[int, int],
                 ):
        """
        @mesh is normalized to have center around (0, 0, 0)
        vol_whl: mesh width height length (slighly enlarged)
        """
        self.f = f
        self.vol_whl = vol_whl
        self.batch_size = batch_size
        self.rand_std = rand_std

    def __len__(self):
        return 5000 # TODO: modify this according to mesh complexity

    def __getitem__(self, idx: int):
        """
        50% points near the surface + gaussian
        50% uniform in the bounding volume
        """
        uniform_pts = np.random.random((self.batch_size//2, 3))
        uniform_pts = (uniform_pts-0.5)*2*self.vol_whl
        surface_pts = self.f.sample_surface(self.batch_size//2)
        surface_pts += self.rand_std*self.vol_whl* \
                       np.random.randn(*surface_pts.shape)

        xyz = np.concatenate([uniform_pts, surface_pts])
        occ = self.f.contains(xyz)

        xyz = torch.FloatTensor(xyz)
        occ = torch.FloatTensor(occ)

        return {'xyz': xyz, 'occ': occ, 'occ_ratio': occ.mean()}