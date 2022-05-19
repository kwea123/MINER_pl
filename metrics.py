import torch
import numpy as np


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return value.mean()
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    if torch.is_tensor(image_pred):
        return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))
    elif isinstance(image_pred, np.ndarray):
        return -10*np.log10(mse(image_pred, image_gt, valid_mask, reduction))


def iou(occ1, occ2):
    occ1 = occ1>=0.5
    occ2 = occ2>=0.5

    area_union = (occ1 | occ2).sum()
    area_intersect = (occ1 & occ2).sum()

    return area_intersect/(area_union+1e-8)