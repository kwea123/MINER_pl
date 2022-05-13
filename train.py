import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import cv2

from opt import get_opts

# datasets
from dataset import ImageDataset, collate_fn
from torch.utils.data import DataLoader

# models
from models import BlockMLP

# metrics
from metrics import psnr

# optimizer
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def reshape_image(image, hparams):
    # reshape image for visualization
    return rearrange(image, '(nh nw) (ph pw) c -> 1 c (nh ph) (nw pw)',
                     nh=hparams.nh,
                     nw=hparams.nw,
                     ph=hparams.patch_wh[1],
                     pw=hparams.patch_wh[0])


class MINERSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.first_val = True

        self.blockmlp = BlockMLP(n_blocks=hparams.n_blocks, 
                                 n_in=2, n_out=3,
                                 n_layers=hparams.n_layers,
                                 n_hidden=hparams.n_hidden,
                                 final_act=hparams.final_act,
                                 a=hparams.a)
        self.register_buffer('training_blocks',
                             torch.ones(hparams.n_blocks, dtype=torch.bool))

    def forward(self, x, training_blocks, b_chunk):
        out = self.blockmlp(x, training_blocks, b_chunk)
        if hparams.level<=hparams.n_scales-2 and hparams.pyr=='laplacian':
            if training_blocks is None:
                out *= self.scales
            else:
                out *= self.scales[training_blocks]
        return out
        
    def setup(self, stage=None):
        global I_j_gt
        # TODO: use different dataset for train and val
        # only active blocks for train, and all blocks for val
        # currently rgb is always all blocks
        self.dataset = ImageDataset(I_j_gt,
                                    hparams.img_wh,
                                    hparams.patch_wh,
                                    hparams.n_blocks)

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=collate_fn,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.opt = RAdam(self.blockmlp.parameters(), lr=hparams.lr)
        sch = CosineAnnealingLR(self.opt,
                                hparams.num_epochs,
                                hparams.lr/30)

        return [self.opt], [sch]

    def training_step(self, batch, batch_idx):
        if self.first_val: # some trick to log the values after val_sanity_check
            global psnr_, n_training_blocks
            self.log('val/psnr', psnr_, True,
                     on_step=False, on_epoch=True)
            self.log('val/n_training_blocks', n_training_blocks, True,
                     on_step=False, on_epoch=True)
            self.first_val = False

        loss = self(batch['uv'], self.training_blocks, hparams.b_chunk) - \
               batch['rgb'][self.active_blocks][self.training_blocks]
        loss = (loss**2).mean()

        self.log('lr', get_learning_rate(self.opt))
        self.log('train/loss', loss)

        return loss

    def on_validation_start(self):
        global params_dict
        if not self.first_val:
            # reset the parameters back for freezed blocks
            with torch.no_grad():
                for n, p in self.blockmlp.named_parameters():
                    p[~self.training_blocks] = params_dict[n][~self.training_blocks]

    def validation_step(self, batch, batch_idx):
        return {'rgb_gt': batch['rgb'],
                'rgb_pred': self(batch['uv'], None, hparams.b_chunk)}

    def validation_epoch_end(self, outputs):
        global I_j_u_, psnr_, n_training_blocks
        rgb_gt = torch.cat([x['rgb_gt'] for x in outputs], 1) # always all blocks
        rgb_pred = torch.cat([x['rgb_pred'] for x in outputs], 1) # depends on active blocks

        # remove converged blocks
        loss = reduce((rgb_pred-rgb_gt[self.active_blocks])**2,
                      'n p c -> n', 'mean')
        self.training_blocks = loss>hparams.loss_thr
        n_training_blocks = self.training_blocks.sum().float()
        self.log('val/n_training_blocks', n_training_blocks, True)

        # visualization of active blocks, rgb
        rgb_pred_ = torch.zeros_like(rgb_gt)
        rgb_pred_[self.active_blocks] = rgb_pred
        if hparams.level<=hparams.n_scales-2:
            if hparams.pyr=='gaussian':
                rgb_pred_[~self.active_blocks] = I_j_u_[~self.active_blocks]
            elif hparams.pyr=='laplacian':
                lap_gt = reshape_image(rgb_gt, hparams)
                lap_pred = reshape_image(rgb_pred_, hparams)
                self.logger.experiment.add_images(
                    f'laplacian/l{hparams.level}',
                    torch.cat([(lap_gt+1)/2, (lap_pred+1)/2]), # normalize to [0, 1]
                    self.global_step)
                # add upsampled image to laplacian
                rgb_gt += I_j_u_
                rgb_pred_ += I_j_u_
            rgb_pred = rgb_pred_

        self.rgb_pred = reshape_image(rgb_pred, hparams)
        rgb_gt = reshape_image(rgb_gt, hparams)
        self.rgb_pred = torch.clip(self.rgb_pred, 0, 1)
        rgb_gt = torch.clip(rgb_gt, 0, 1)

        if hparams.level<=hparams.n_scales-2:
            if self.first_val: # log this only once (doesn't change during training)
                active_blocks_v = \
                    rearrange(self.active_blocks, '(nh nw) -> 1 nh nw',
                              nh=hparams.nh,
                              nw=hparams.nw)
                active_blocks_v = \
                    repeat(active_blocks_v, '1 nh nw -> 1 (nh ph) (nw pw)',
                           ph=hparams.patch_wh[1],
                           pw=hparams.patch_wh[0])
                self.logger.experiment.add_image(f'active_blocks/l{hparams.level}',
                                                 (rgb_gt[0]+active_blocks_v)/2,
                                                 self.global_step)
            
        self.logger.experiment.add_images(f'image/l{hparams.level}',
                                          torch.cat([rgb_gt, self.rgb_pred]),
                                          self.global_step)

        psnr_ = psnr(self.rgb_pred, rgb_gt)
        self.log('val/psnr', psnr_, True)

    def on_validation_end(self):
        global params_dict
        params_dict = {}
        for n, p in self.blockmlp.named_parameters():
            params_dict[n] = p.clone() # save params


if __name__ == '__main__':
    hparams = get_opts()
    assert hparams.img_wh[0]%(hparams.patch_wh[0]*2**(hparams.n_scales-1))==0 and \
           hparams.img_wh[1]%(hparams.patch_wh[1]*2**(hparams.n_scales-1))==0, \
           'img_wh must be a multiple of patch_wh*2**(n_scales-1)!'
    hparams.batch_size = min(hparams.batch_size,
                             hparams.patch_wh[0]*hparams.patch_wh[1])
    num_epochs = hparams.num_epochs[::-1]

    # load image of the original scale once
    image = np.float32(Image.open(hparams.image_path).convert('RGB'))/255.

    # train n_scales progressively.
    for j in reversed(range(hparams.n_scales)): # J-1 ~ 0 coarse to fine
        hparams.level = j
        hparams.final_act = 'sigmoid'

        I_j_gt = cv2.resize(image,
                    (hparams.img_wh[0]//(2**j), hparams.img_wh[1]//(2**j)))
        # number of horizontal and vertical blocks
        hparams.nh = hparams.img_wh[1]//(hparams.patch_wh[1]*2**j)
        hparams.nw = hparams.img_wh[0]//(hparams.patch_wh[0]*2**j)
        if j <= hparams.n_scales-2:
            I_j_u_ = rearrange(I_j_u,
                               '1 c (nh ph) (nw pw) -> (nh nw) (ph pw) c',
                               nh=hparams.nh,
                               nw=hparams.nw)
            I_j_gt_ = rearrange(I_j_gt, '(nh ph) (nw pw) c -> (nh nw) (ph pw) c',
                                nh=hparams.nh,
                                nw=hparams.nw)
            I_j_gt_ = torch.tensor(I_j_gt_, dtype=I_j_u.dtype, device=I_j_u.device)
            residual = I_j_gt_-I_j_u_
            # compute active blocks
            loss = reduce(residual**2, 'n p c -> n', 'mean')
            active_blocks = loss>hparams.loss_thr
            hparams.n_blocks = active_blocks.sum().item()
            if hparams.pyr=='laplacian': # compute residual
                scales = reduce(torch.abs(residual[active_blocks]),
                                'n p c -> n 1 1', 'max')
                I_j_gt -= rearrange(I_j_u.cpu().numpy(), '1 c h w -> h w c')
                hparams.final_act = 'sin'
        else: # coarsest level
            hparams.n_blocks = hparams.nh*hparams.nw
            active_blocks = torch.ones(hparams.n_blocks, dtype=torch.bool)

        system = MINERSystem(hparams)
        system.register_buffer("active_blocks", active_blocks)
        if j<=hparams.n_scales-2 and hparams.pyr=='laplacian':
            system.register_buffer("scales", scales)

        ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}/l{j}',
                                  save_last=True,
                                  save_top_k=0,
                                  save_weights_only=True)
        pbar = TQDMProgressBar(refresh_rate=1)
        callbacks = [ckpt_cb, pbar]

        logger = TensorBoardLogger(save_dir='logs',
                                   name=f'{hparams.exp_name}/l{j}',
                                   default_hp_metric=False)

        hparams.num_epochs = num_epochs[min(j, len(num_epochs)-1)]
        trainer = Trainer(max_epochs=hparams.num_epochs,
                          callbacks=callbacks,
                          logger=logger,
                          enable_model_summary=True,
                          accelerator='auto',
                          devices=1,
                          num_sanity_val_steps=-1, # validate the whole image once before training
                          log_every_n_steps=1,
                          check_val_every_n_epoch=hparams.val_freq,
                          benchmark=True)

        trainer.fit(system)

        # upsample the predicted image for the next level
        I_j_u = F.interpolate(system.rgb_pred,
                              mode='bilinear',
                              scale_factor=2,
                              align_corners=True)
