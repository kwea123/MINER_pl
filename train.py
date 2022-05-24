import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce, repeat
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # enable reading large image
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings("ignore")

from opt import get_opts

# datasets
from dataset import CoordinateDataset
from torch.utils.data import DataLoader

# models
from models import E_2d, E_3d, PE, BlockMLP, BlockMLP_Gabor
from patterns import patterns_dict, einops_f

# metrics
from metrics import mse, psnr, iou

# optimizer
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


class MINERSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.first_val = True
        self.automatic_optimization = False

        if hparams.task=='image':
            n_in = 2 # uv
            n_out = 3 # rgb
        elif hparams.task=='mesh':
            n_in = 3 # xyz
            n_out = 1 # occ

        if hparams.use_pe:
            if hparams.task=='image': E = E_2d
            elif hparams.task=='mesh': E = E_3d
            P = torch.cat([E*2**i for i in range(hparams.n_freq)], 1)
            self.pe = PE(P)
            n_in = self.pe.out_dim

        # create two copies of the same network
        # the network used in training
        if hparams.arch == 'mlp':
            self.optim = RAdam
            self.blockmlp_ = BlockMLP(
                                n_blocks=hparams.n_blocks, 
                                n_in=n_in, n_out=n_out,
                                n_layers=hparams.n_layers,
                                n_hidden=hparams.n_hidden,
                                final_act=hparams.final_act,
                                a=hparams.a)
        elif hparams.arch == 'gabor':
            self.optim = Adam
            self.blockmlp_ = BlockMLP_Gabor(
                                n_blocks=hparams.n_blocks, 
                                n_in=n_in, n_out=n_out,
                                n_layers=hparams.n_layers,
                                n_hidden=hparams.n_hidden,
                                final_act=hparams.final_act,
                                a=hparams.a,
                                weight_scale=32.0)
        # the network used in validation, updated by the trained network
        self.blockmlp = copy.deepcopy(self.blockmlp_)
        for p in self.blockmlp.parameters():
            p.requires_grad = False

        self.register_buffer('training_blocks',
                             torch.ones(hparams.n_blocks, dtype=torch.bool))

    def call(self, model, x, b_chunks):
        kwargs = {'to_cpu': not self.blockmlp.training}
        if hparams.use_pe: kwargs['pe'] = self.pe
        out = model(x, b_chunks, **kwargs)
        if hparams.level<=hparams.n_scales-2 and hparams.pyr=='laplacian':
            if self.blockmlp.training:
                out *= self.scales[self.training_blocks]
            else:
                out *= self.scales.cpu()
        return out
        
    def setup(self, stage=None):
        # validation is always the whole data
        self.val_dataset = CoordinateDataset(self.I_j_gt, hparams)

    def train_dataloader(self):
        # load only active blocks to accelerate
        active_blocks = self.active_blocks.clone()
        active_blocks[self.active_blocks] = self.training_blocks
        train_dataset = CoordinateDataset(
                            self.I_j_gt,
                            hparams,
                            active_blocks.cpu())
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
        self.opt = self.optim(self.blockmlp_.parameters(), lr=hparams.lr)
        self.sch = CosineAnnealingLR(self.opt,
                                     hparams.num_epochs,
                                     hparams.lr/30)

    def training_step(self, batch, batch_idx):
        if self.first_val: # some trick to log the values after val_sanity_check
            if hparams.task == 'image':
                self.log('val/psnr', self.psnr_, True,
                         on_step=False, on_epoch=True)
            elif hparams.task == 'mesh':
                self.log('val/iou', self.iou, True,
                         on_step=False, on_epoch=True)
            self.log('val/n_training_blocks', self.n_training_blocks, True,
                     on_step=False, on_epoch=True)
            self.first_val = False

        inp = einops_f(batch['inp'], 'b n c -> n b c')
        gt = einops_f(batch['out'], 'b n c -> n b c')
        pred = self.call(self.blockmlp_, inp, hparams.b_chunks)
        mse_ = mse(pred, gt, reduction='none')
        loss = reduce(mse_, 'n p c -> n', 'mean')
        # heuristics: easier blocks have higher weights (make them converge faster)
        weight = 1/(loss.detach()+1e-8)

        self.opt_.zero_grad()
        self.manual_backward((weight*loss).mean())
        self.opt_.step()

        self.log('lr', self.opt_.param_groups[0]['lr'])
        self.log('train/loss', mse_.mean(), True)

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
        inp = einops_f(batch['inp'], 'p n c -> n p c')
        inp = repeat(inp, '1 p c -> n p c', n=int(self.active_blocks.sum()))
        gt = einops_f(batch['out'], 'p n c -> n p c')
        pred = self.call(self.blockmlp, inp, hparams.b_chunks)

        return {'gt': gt, 'pred': pred}

    def validation_epoch_end(self, outputs):
        gt = torch.cat([x['gt'] for x in outputs], 1).cpu() # always all blocks
        pred = torch.cat([x['pred'] for x in outputs], 1) # depends on active blocks

        # remove converged blocks
        active_blocks_cpu = self.active_blocks.cpu()
        mse_ = mse(pred, gt[active_blocks_cpu], reduction='none')
        loss = reduce(mse_, 'n p c -> n', 'mean')
        training_blocks_cpu = loss>hparams.loss_thr
        self.training_blocks = training_blocks_cpu.to(self.training_blocks.device)
        self.n_training_blocks = self.training_blocks.sum().float()
        self.log('val/n_training_blocks', self.n_training_blocks, True)

        # visualize training blocks
        tb = self.logger.experiment
        if hparams.level<=hparams.n_scales-2:
            rgb_pred_ = torch.zeros_like(gt)
            rgb_pred_[active_blocks_cpu] = pred
            if hparams.pyr=='gaussian':
                rgb_pred_[~active_blocks_cpu] = self.I_j_u_[~active_blocks_cpu]
            elif hparams.pyr=='laplacian':
                if hparams.task=='image' and hparams.log_image:
                    lap_gt = einops_f(gt, hparams.patterns['reshape'][4], hparams)
                    lap_pred = einops_f(rgb_pred_, hparams.patterns['reshape'][4],
                                       hparams)
                    tb.add_images(f'laplacian/l{hparams.level}',
                                  torch.cat([(lap_gt+1)/2, (lap_pred+1)/2]),
                                  self.current_epoch)
                # add upsampled pred to laplacian
                gt += self.I_j_u_
                rgb_pred_ += self.I_j_u_
            pred = rgb_pred_

        self.pred = torch.clip(einops_f(pred, hparams.patterns['reshape'][4], hparams), 0, 1)
        gt = torch.clip(einops_f(gt, hparams.patterns['reshape'][4], hparams), 0, 1)

        if hparams.log_image and hparams.task=='image':
            blocks = active_blocks_cpu.clone()
            if not self.first_val:
                blocks[active_blocks_cpu] = training_blocks_cpu
            blocks_v = einops_f(blocks, hparams.patterns['reshape'][5], hparams)
            blocks_v = einops_f(blocks_v, hparams.patterns['reshape'][6],
                                hparams, repeat)
            tb.add_image(f'training_blocks/l{hparams.level}',
                         (gt[0]+blocks_v)/2,
                         self.current_epoch)
            tb.add_images(f'image/l{hparams.level}',
                          torch.cat([gt, self.pred]),
                          self.current_epoch)

        if hparams.task == 'image':
            self.psnr_ = psnr(self.pred, gt)
            self.log('val/psnr', self.psnr_, True)
        elif hparams.task == 'mesh':
            self.iou = iou(self.pred, gt)
            self.log('val/iou', self.iou, True)

    def on_validation_end(self):
        # save checkpoint
        ckpt_path = f'ckpts/{hparams.exp_name}'
        os.makedirs(ckpt_path, exist_ok=True)
        state_dict = self.blockmlp.state_dict()
        state_dict['active_blocks'] = self.active_blocks
        state_dict['training_blocks'] = self.training_blocks
        if hparams.level <= hparams.n_scales-2:
            state_dict['scales'] = self.scales
        torch.save(state_dict, f'{ckpt_path}/l{j}.ckpt')

        # create new blockmlp_ with reduced blocks
        for n, p in self.blockmlp.named_parameters():
            setattr(self.blockmlp_, n, nn.Parameter(p[self.training_blocks].data))

        # create new opt_ with reduced blocks
        self.opt_ = self.optim(self.blockmlp_.parameters(),
                               lr=self.sch.get_last_lr()[0])
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
    # support 1 int arguments
    if len(hparams.input_size)==1:
        if hparams.task=='image':
            hparams.input_size = tuple(hparams.input_size[0] for _ in range(2))
        elif hparams.task=='mesh':
            hparams.input_size = tuple(hparams.input_size[0] for _ in range(3))
    if len(hparams.patch_size)==1:
        if hparams.task=='image':
            hparams.patch_size = tuple(hparams.patch_size[0] for _ in range(2))
        elif hparams.task=='mesh':
            hparams.patch_size = tuple(hparams.patch_size[0] for _ in range(3))
    assert all(hparams.input_size[i]%(hparams.patch_size[i]*2**(hparams.n_scales-1))==0 
                for i in range(len(hparams.input_size))), \
        'input_size must be a multiple of patch_size*2**(n_scales-1)!'
    assert hparams.num_epochs[-1]%hparams.val_freq==0, \
        'last num_epochs must be a multiple of val_freq!'
    for i in range(len(hparams.patch_size)):
        setattr(hparams, f'p{i+1}', hparams.patch_size[i])
    hparams.batch_size = min(hparams.batch_size, np.prod(hparams.patch_size))
    num_epochs = hparams.num_epochs[::-1]

    hparams.patterns = patterns_dict[hparams.task]

    # load input and resize to input_size
    print('loading input ...')
    if hparams.task == 'image':
        inp = np.float32(Image.open(hparams.path).convert('RGB'))/255.
    elif hparams.task == 'mesh':
        # precomputed (N, N, N, 1) occupancies
        inp = np.unpackbits(np.load(hparams.path)) \
              .reshape(*hparams.input_size)[..., None].astype(np.float32)
    inp = einops_f(inp, hparams.patterns['reshape'][0])
    inp = F.interpolate(torch.from_numpy(inp),
                        size=hparams.input_size[::-1],
                        mode=hparams.patterns['mode'],
                        align_corners=True)
    print('input loaded!')

    # train n_scales progressively
    for j in reversed(range(hparams.n_scales)): # J-1 ~ 0 coarse to fine
        hparams.level = j
        hparams.final_act = 'sigmoid'

        if j==0:
            I_j_gt = inp
        else:
            I_j_gt = F.interpolate(inp,
                                   mode=hparams.patterns['mode'],
                                   scale_factor=1/2**j,
                                   align_corners=True)
        I_j_gt = einops_f(I_j_gt, hparams.patterns['reshape'][1])

        # compute number of blocks in each dimension
        n_blocks = 1
        for i in range(len(hparams.input_size)):
            ni = hparams.input_size[i]//(hparams.patch_size[i]*2**j)
            setattr(hparams, f'n{i+1}', ni)
            n_blocks *= ni

        if j<=hparams.n_scales-2:
            I_j_u_ = einops_f(I_j_u, hparams.patterns['reshape'][2], hparams)
            I_j_gt_ = einops_f(I_j_gt, hparams.patterns['reshape'][3], hparams)
            I_j_gt_ = torch.tensor(I_j_gt_, dtype=I_j_u.dtype, device=I_j_u.device)
            residual = I_j_gt_-I_j_u_
            # compute active blocks
            loss = reduce(residual**2, 'n p c -> n', 'mean')
            active_blocks = loss>hparams.loss_thr
            hparams.n_blocks = active_blocks.sum().item()
            if hparams.pyr=='laplacian': # compute residual
                scales = reduce(torch.abs(residual[active_blocks]),
                                'n p c -> n 1 1', 'max')
                I_j_gt -= einops_f(I_j_u.cpu().numpy(),
                                   hparams.patterns['reshape'][1])
                hparams.final_act = 'sin'
            del I_j_gt_, residual, loss
        else: # coarsest level
            hparams.n_blocks = n_blocks
            active_blocks = torch.ones(hparams.n_blocks, dtype=torch.bool)

        system = MINERSystem(hparams)
        system.register_buffer("active_blocks", active_blocks)
        system.I_j_gt = I_j_gt
        del I_j_gt
        if j<=hparams.n_scales-2 and hparams.pyr=='laplacian':
            system.I_j_u_ = I_j_u_
            del I_j_u_
            system.register_buffer("scales", scales)

        logger = TensorBoardLogger(save_dir='logs',
                                   name=f'{hparams.exp_name}/l{j}',
                                   default_hp_metric=False)
        callbacks = [TQDMProgressBar(refresh_rate=1),
                     EarlyStopping('val/n_training_blocks',
                                   stopping_threshold=0.1)] # stop training if n_training_blocks reaches 0

        hparams.num_epochs = num_epochs[min(j, len(num_epochs)-1)]
        trainer = Trainer(max_epochs=hparams.num_epochs,
                          callbacks=callbacks,
                          logger=logger,
                          enable_model_summary=True,
                          accelerator='auto',
                          devices=1,
                          num_sanity_val_steps=-1, # validate the whole data once before training
                          log_every_n_steps=1,
                          reload_dataloaders_every_n_epochs=1,
                          check_val_every_n_epoch=hparams.val_freq if j==0 else hparams.num_epochs)
        trainer.fit(system)

        del logger, callbacks, trainer
        if j>0: # upsample the pred for the next level
            I_j_u = F.interpolate(system.pred,
                                  mode=hparams.patterns['mode'],
                                  scale_factor=2,
                                  align_corners=True)
        else:
            pred = system.pred
            del system

    if hparams.task == 'image':
        psnr_ = psnr(pred.cpu(), inp).numpy()
        print(f'PSNR : {psnr_:.4f} dB')
    elif hparams.task == 'mesh':
        iou_ = iou(pred.cpu(), inp).numpy()
        print(f'IoU : {iou_:.6f}')
