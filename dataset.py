import torch
from torch.utils.data import Dataset
from kornia import create_meshgrid
from einops import rearrange, repeat
import numpy as np
from typing import Tuple


def collate_fn(batch):
    uv, rgb = [], []
    for data in batch:
        uv += [data['uv']]
        rgb += [data['rgb']]
    return {'uv': torch.stack(uv, 1),
            'rgb': torch.stack(rgb, 1)}


class ImageDataset(Dataset):
    def __init__(self,
                 image: np.ndarray,
                 img_wh: Tuple[int, int],
                 patch_wh: Tuple[int, int],
                 n_blocks: int):
        """
        image: image resized to the current level
               subtracted by the upsampled reconstruction of the previous level
               in finer levels
        """
        self.img_wh = img_wh
        self.patch_wh = patch_wh

        # split into patches
        image = rearrange(image,
                          '(nh ph) (nw pw) c -> (nh nw) (ph pw) c',
                          ph=patch_wh[1], pw=patch_wh[0])
        self.rgb = torch.tensor(image)

        self.uv = create_meshgrid(patch_wh[1], patch_wh[0])
        self.uv = rearrange(self.uv, '1 ph pw c -> 1 (ph pw) c')
        self.uv = repeat(self.uv, '1 phw c -> n phw c', n=n_blocks)

    def __len__(self):
        return self.patch_wh[0]*self.patch_wh[1]

    def __getitem__(self, idx: int):
        # each batch contains all blocks with randomly selected pixels
        # the pixels in each block are of the same position
        # TODO: Choosing position randomly for each block independently
        # might lead to better performance...
        # TODO: Random select some blocks only instead of all blocks
        return {"uv": self.uv[:, idx],
                "rgb": self.rgb[:, idx]}