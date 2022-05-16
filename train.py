import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # enable reading large image
import numpy as np
import cv2
import copy
import os
import warnings
warnings.filterwarnings("ignore")

from opt import get_opts

# datasets
from dataset import ImageDataset
from torch.utils.data import DataLoader

# models
from models import PE, BlockMLP

# metrics
from metrics import psnr

# optimizer
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


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
        self.automatic_optimization = False

        if hparams.use_pe:
            P = torch.cat([torch.eye(2)*2**i for i in range(hparams.n_freq)], 1)
            self.pe = PE(P)
            n_in = self.pe.out_dim
        else:
            n_in = 2

        # create two copies of the same network
        # the network used in training
        self.blockmlp_ = BlockMLP(n_blocks=hparams.n_blocks, 
                                  n_in=n_in, n_out=3,
                                  n_layers=hparams.n_layers,
                                  n_hidden=hparams.n_hidden,
                                  final_act=hparams.final_act,
                                  a=hparams.a)
        # the network used in validation, updated by the trained network
        self.blockmlp = copy.deepcopy(self.blockmlp_)

        self.register_buffer('training_blocks',
                             torch.ones(hparams.n_blocks, dtype=torch.bool))

    def forward(self, model, x, b_chunks):
        if hparams.use_pe:
            x = self.pe(x)
        out = model(x, b_chunks, not self.blockmlp.training)
        if hparams.level<=hparams.n_scales-2 and hparams.pyr=='laplacian':
            if self.blockmlp.training:
                out *= self.scales[self.training_blocks]
            else:
                out *= self.scales.cpu()
        return out
        
    def setup(self, stage=None):
        # validation is always the whole image
        self.val_dataset = ImageDataset(self.I_j_gt,
                                        hparams.subimg_wh,
                                        hparams.patch_wh)

    def train_dataloader(self):
        # load only active blocks to accelerate
        active_blocks = self.active_blocks_cpu.clone()
        active_blocks[self.active_blocks_cpu] = self.training_blocks_cpu
        train_dataset = ImageDataset(self.I_j_gt,
                                     hparams.subimg_wh,
                                     hparams.patch_wh,
                                     active_blocks)
        return DataLoader(train_dataset,
                          shuffle=True,
                          num_workers=0,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        # dummy that stores optimizer states and performs scheduling
        # real optimizers are defined in @on_validation_end
        self.opt = RAdam(self.blockmlp_.parameters(), lr=hparams.lr)
        self.sch = CosineAnnealingLR(self.opt,
                                     hparams.num_epochs,
                                     hparams.lr/30)

    def training_step(self, batch, batch_idx):
        if self.first_val: # some trick to log the values after val_sanity_check
            self.log('val/psnr', self.psnr_, True,
                     on_step=False, on_epoch=True)
            self.log('val/n_training_blocks', self.n_training_blocks, True,
                     on_step=False, on_epoch=True)
            self.first_val = False

        uv = rearrange(batch['uv'], 'p 1 c -> 1 p c')
        uv = repeat(uv, '1 p c -> n p c', n=int(self.n_training_blocks))
        rgb_gt = rearrange(batch['rgb'], 'p n c -> n p c')
        rgb_pred = self(self.blockmlp_, uv, hparams.b_chunk)
        mse = (rgb_pred-rgb_gt)**2
        loss = reduce(mse, 'n p c -> n', 'mean')
        # heuristics: easier blocks have higher weights (make them converge faster)
        weight = 1/(loss.detach()+1e-8)

        self.opt_.zero_grad()
        self.manual_backward((weight*loss).mean())
        self.opt_.step()

        self.log('lr', self.opt_.param_groups[0]['lr'])
        self.log('train/loss', mse.mean(), True)

        if self.trainer.is_last_batch:
            # update opt_'s lr by the scheduler
            self.sch.step()
            self.opt_.param_groups[0]['lr'] = self.sch.get_last_lr()[0]

    def on_validation_start(self):
        if not self.first_val:
            # copy blockmlp weight from blockmlp_
            for p, p_ in zip(self.blockmlp.parameters(),
                             self.blockmlp_.parameters()):
                p.data[self.training_blocks] = p_.data

            # copy opt states from opt_
            for p, p_ in zip(self.opt.param_groups[0]['params'],
                             self.opt_.param_groups[0]['params']):
                for k, v in self.opt_.state[p_].items():
                    if torch.is_tensor(v): # exp_avg, etc
                        if k not in self.opt.state[p]:
                            # Lazy state initialization
                            # ref: https://github.com/pytorch/pytorch/blob/master/torch/optim/radam.py#L117-L123
                            self.opt.state[p][k] = torch.zeros_like(p)
                        self.opt.state[p][k][self.training_blocks] = v
                    else: # step
                        self.opt.state[p][k] = v

    def validation_step(self, batch, batch_idx):
        uv = rearrange(batch['uv'], 'p 1 c -> 1 p c')
        uv = repeat(uv, '1 p c -> n p c', n=int(self.active_blocks.sum()))
        rgb_gt = rearrange(batch['rgb'], 'p n c -> n p c')
        return {'rgb_gt': rgb_gt,
                'rgb_pred': self(self.blockmlp, uv, hparams.b_chunk)}

    def validation_epoch_end(self, outputs):
        rgb_gt = torch.cat([x['rgb_gt'] for x in outputs], 1).cpu() # always all blocks
        rgb_pred = torch.cat([x['rgb_pred'] for x in outputs], 1) # depends on active blocks

        # remove converged blocks
        self.active_blocks_cpu = self.active_blocks.cpu()
        loss = reduce((rgb_pred-rgb_gt[self.active_blocks_cpu])**2,
                      'n p c -> n', 'mean')
        self.training_blocks_cpu = loss>hparams.loss_thr
        self.training_blocks = self.training_blocks_cpu.to(self.training_blocks.device)
        self.n_training_blocks = self.training_blocks.sum().float()
        self.log('val/n_training_blocks', self.n_training_blocks, True)

        # visualize training blocks, rgb
        if hparams.level<=hparams.n_scales-2:
            rgb_pred_ = torch.zeros_like(rgb_gt)
            rgb_pred_[self.active_blocks_cpu] = rgb_pred
            if hparams.pyr=='gaussian':
                rgb_pred_[~self.active_blocks_cpu] = \
                    I_j_u_[~self.active_blocks_cpu]
            elif hparams.pyr=='laplacian':
                if hparams.log_image:
                    lap_gt = reshape_image(rgb_gt, hparams)
                    lap_pred = reshape_image(rgb_pred_, hparams)
                    self.logger.experiment.add_images(
                        f'laplacian/l{hparams.level}',
                        torch.cat([(lap_gt+1)/2, (lap_pred+1)/2]),
                        self.current_epoch)
                # add upsampled image to laplacian
                rgb_gt += I_j_u_
                rgb_pred_ += I_j_u_
            rgb_pred = rgb_pred_

        self.rgb_pred = torch.clip(reshape_image(rgb_pred, hparams), 0, 1)
        rgb_gt = torch.clip(reshape_image(rgb_gt, hparams), 0, 1)

        if hparams.log_image:
            blocks = self.active_blocks_cpu.clone()
            if not self.first_val:
                blocks[self.active_blocks_cpu] = self.training_blocks_cpu
            blocks_v = rearrange(blocks, '(nh nw) -> 1 nh nw',
                                 nh=hparams.nh,
                                 nw=hparams.nw)
            blocks_v = repeat(blocks_v, '1 nh nw -> 1 (nh ph) (nw pw)',
                              ph=hparams.patch_wh[1],
                              pw=hparams.patch_wh[0])
            self.logger.experiment.add_image(f'training_blocks/l{hparams.level}',
                                             (rgb_gt[0]+blocks_v)/2,
                                             self.current_epoch)
            self.logger.experiment.add_images(f'image/l{hparams.level}',
                                              torch.cat([rgb_gt, self.rgb_pred]),
                                              self.current_epoch)

        self.psnr_ = psnr(self.rgb_pred, rgb_gt)
        self.log('val/psnr', self.psnr_, True)

    def on_validation_end(self):
        # save checkpoint
        ckpt_path = f'ckpts/{hparams.exp_name}/subimg{hparams.subimg_idx:03d}'
        os.makedirs(ckpt_path, exist_ok=True)
        state_dict = self.blockmlp.state_dict()
        state_dict['active_blocks'] = self.active_blocks
        state_dict['training_blocks'] = self.training_blocks
        torch.save(state_dict, f'{ckpt_path}/l{j}.ckpt')

        # create new blockmlp_ with reduced blocks
        for n, p in self.blockmlp.named_parameters():
            setattr(self.blockmlp_, n, nn.Parameter(p[self.training_blocks].data))

        # create new opt_ with reduced blocks
        self.opt_ = RAdam(self.blockmlp_.parameters(), lr=self.sch.get_last_lr()[0])
        if not self.first_val:
            # inherit the states: step, exp_avg, etc
            for p, p_ in zip(self.opt.param_groups[0]['params'],
                             self.opt_.param_groups[0]['params']):
                for k, v in self.opt.state[p].items():
                    if torch.is_tensor(v):
                        self.opt_.state[p_][k] = v[self.training_blocks]
                    else:
                        self.opt_.state[p_][k] = v


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.img_wh[0]%hparams.subimg_wh[0]!=0 or \
       hparams.img_wh[1]%hparams.subimg_wh[1]!=0:
        print('subimg_wh is wrong! Setting it to img_wh ...')
        hparams.subimg_wh = hparams.img_wh
    assert hparams.subimg_wh[0]%(hparams.patch_wh[0]*2**(hparams.n_scales-1))==0 and \
           hparams.subimg_wh[1]%(hparams.patch_wh[1]*2**(hparams.n_scales-1))==0, \
           'subimg_wh must be a multiple of patch_wh*2**(n_scales-1)!'
    assert hparams.num_epochs[-1]%hparams.val_freq==0, \
           'last num_epochs must be a multiple of val_freq!'
    hparams.batch_size = min(hparams.batch_size,
                             hparams.patch_wh[0]*hparams.patch_wh[1])
    num_epochs = hparams.num_epochs[::-1]

    # load image of the original scale once
    orig_image = np.float32(Image.open(hparams.image_path).convert('RGB'))/255.
    orig_image = cv2.resize(orig_image, (hparams.img_wh[0], hparams.img_wh[1]))

    # split into sub-images
    nw_sub = hparams.img_wh[0]//hparams.subimg_wh[0]
    nh_sub = hparams.img_wh[1]//hparams.subimg_wh[1]
    sub_images = rearrange(orig_image,
                           '(nh ph) (nw pw) c -> (nh nw) ph pw c',
                           nh=nh_sub, nw=nw_sub,
                           ph=hparams.subimg_wh[1], pw=hparams.subimg_wh[0])
    del orig_image

    sub_psnrs = []

    # train each sub-image independently
    for i in range(nw_sub*nh_sub):
        image = sub_images[0]
        sub_images = sub_images[1:] # pop the first image and remove it to save memory
        # train n_scales progressively
        for j in reversed(range(hparams.n_scales)): # J-1 ~ 0 coarse to fine
            hparams.subimg_idx = i
            hparams.level = j
            hparams.final_act = 'sigmoid'

            I_j_gt = cv2.resize(image,
                        (hparams.subimg_wh[0]//(2**j), hparams.subimg_wh[1]//(2**j)))
            # number of horizontal and vertical blocks
            hparams.nh = hparams.subimg_wh[1]//(hparams.patch_wh[1]*2**j)
            hparams.nw = hparams.subimg_wh[0]//(hparams.patch_wh[0]*2**j)
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
            system.I_j_gt = I_j_gt
            if j<=hparams.n_scales-2 and hparams.pyr=='laplacian':
                system.I_j_u_ = I_j_u_
                system.register_buffer("scales", scales)

            pbar = TQDMProgressBar(refresh_rate=1)
            callbacks = [pbar]

            logger = TensorBoardLogger(save_dir='logs',
                                       name=f'{hparams.exp_name}/subimg{i:03d}/l{j}',
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
                              reload_dataloaders_every_n_epochs=1,
                              check_val_every_n_epoch=hparams.val_freq if j==0 else hparams.num_epochs)
            trainer.fit(system)

            # upsample the predicted image for the next level
            I_j_u = F.interpolate(system.rgb_pred,
                                  mode='bilinear',
                                  scale_factor=2,
                                  align_corners=True)

        # compute the psnr for this sub-image
        sub_psnrs += [psnr(rearrange(system.rgb_pred.numpy(), '1 c h w -> h w c'), image)]

    print(f'PSNR : {np.mean(sub_psnrs):.2f} dB')