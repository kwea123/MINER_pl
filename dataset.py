import torch
from torch.utils.data import Dataset
from kornia import create_meshgrid
from einops import rearrange
import numpy as np
from typing import Tuple


class ImageDataset(Dataset):
    def __init__(self,
                 image: np.ndarray,
                 img_wh: Tuple[int, int],
                 patch_wh: Tuple[int, int],
                 active_blocks=None):
        """
        image: image resized to the current level
               subtracted by the upsampled reconstruction of the previous level
               in finer levels
        active_blocks: np.ndarray, None to return all blocks,
                       otherwise specify the blocks to take
        """
        self.img_wh = img_wh
        self.patch_wh = patch_wh

        # split into patches
        image = rearrange(image,
                          '(nh ph) (nw pw) c -> (nh nw) (ph pw) c',
                          ph=patch_wh[1], pw=patch_wh[0])
        # if active_blocks is not None:
        #     image = image[active_blocks]
        self.rgb = torch.tensor(image)
        if active_blocks is not None:
            self.rgb = self.rgb[active_blocks]

        self.uv = create_meshgrid(patch_wh[1], patch_wh[0])
        self.uv = rearrange(self.uv, '1 ph pw c -> 1 (ph pw) c')

    def __len__(self):
        return self.patch_wh[0]*self.patch_wh[1]

    def __getitem__(self, idx: int):
        # each batch contains all blocks with randomly selected pixels
        # the pixels in each block are of the same position
        return {"uv": self.uv[:, idx], "rgb": self.rgb[:, idx]}