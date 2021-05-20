import sys
import os
import os.path
from sys import exit as e
from os.path import join as pjn
import copy
import glob
import shutil
from datetime import datetime
import math
import hydra
import logging
from tqdm import tqdm
import q

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

import ShanghaiTech_dataset as dtst

from SDCNet import SDCNet
from labels_counts_utils import make_label2count_list
import loss


def print_dbg_info_dataloader(loader):
    """
    Prints image names and tensor shapes for samples in a DataLoader.
    Was used for debugging.
    """
    for i, sample in enumerate(loader):
        print(f"i = {i}; image bname = {sample['image_bname']}; "
              f"image shape = {tuple(sample['image'].shape)}")
        cs = sample['counts_gt']
        ls = sample['labels_gt']
        print(f"counts_gt shapes = {[tuple(l.shape) for l in cs]}; "
              f"labels_gt shapes = {[tuple(l.shape) for l in ls]}")
        print()


def get_dataloaders(cfg, train_val_test_mask):
    """
    Constructs Dataset objects and corresponding DataLoader objects.

    Args:
        cfg: the global configuration (hydra).
        train_val_test_mask: a list or tuple containing three values
            that will be interpreted as booleans, for example (1,1,0).
            The corresponding DataLoaders for the (train, val, test)
            filtered list of Datasets will be created and returned.

    Returns:
        The tuple (train_loader, val_loader, test_loader). The loaders
        corresponding to the `False` values in `train_val_test_mask` 
        are set to `None`.
    """
    interval_bounds, label2count_list = make_label2count_list(cfg)

    assert len(label2count_list) == interval_bounds.shape[0] + 1, \
        "Number of class labels must be equal to (1 + number of interval " \
        "boundaries), but the equality does not hold"

    rgb_mean_train = dtst.calc_rgb_mean_train(cfg)

    composed_transf_train = transforms.Compose([
        dtst.Normalize(rgb_mean_train),
        dtst.ToCHW(),
        dtst.RandomHorizontalFlip(p=0.5),
        dtst.QuasiRandomCrop(),
        dtst.PadToMultipleOf64(),
        dtst.AddGtCountsLabels(interval_bounds),
    ])

    composed_transf_test = transforms.Compose([
        dtst.Normalize(rgb_mean_train),
        dtst.ToCHW(),
        dtst.PadToMultipleOf64(),
    ])

    if train_val_test_mask[0]:
        train_dataset = dtst.ShanghaiTechDataset(
            cfg,
            subdir='train_data',
            shuffle_seed=42,
            rel_inds=(0.0, cfg.train.train_val_split),
            transform=composed_transf_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=4)
    else:
        train_loader = None

    if train_val_test_mask[1]:
        val_dataset = dtst.ShanghaiTechDataset(
            cfg,
            subdir='train_data',
            shuffle_seed=42,
            rel_inds=(cfg.train.train_val_split, 1.0),
            transform=composed_transf_test)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.validation.batch_size,
            shuffle=False,
            num_workers=4)
    else:
        val_loader = None

    if train_val_test_mask[2]:
        test_dataset = dtst.ShanghaiTechDataset(
            cfg,
            subdir='test_data',
            shuffle_seed=None,
            rel_inds=(0.0, 1.0),
            transform=composed_transf_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            num_workers=4)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


class TrainManager(object):
    def __init__(
            self,
            model,
            optimizer,
            cfg,
            additional_cfg,
            train_loader,
            val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.cfg = cfg
        self.add_cfg = additional_cfg
        self.tbx_wrtr_dir = additional_cfg.get('tbx_wrtr_dir')
        self.orig_cwd = hydra.utils.get_original_cwd()
        self.tbx_wrtr = additional_cfg.get('tbx_wrtr')

    def validate(self, data_loader, step=0):
        """
        Run inference for the samples provided by `data_loader`
        and calculate mean average error (MAE) and root-mean-square 
        error (MSE).
        """
        self.model.eval()
        diffs_count_pred_gt = []
        mae_sum = 0.0
        mse_sum = 0.0

        with torch.no_grad():
            for sample in data_loader:
                gt_count = float(sample['num_annot_headpoints'].numpy())
                image = sample['image'].float().to(self.add_cfg['device'])
                *cls_logits_list, DIV2, U1, U2, W1, W2 = self.model(image)
                pred_count = DIV2.cpu().numpy().sum()
                diff = pred_count - gt_count
                diffs_count_pred_gt.append(diff)
                mae_sum += abs(diff)
                mse_sum += diff**2

        N = len(data_loader)
        mae = mae_sum / N
        mse = math.sqrt(mse_sum / N)

        return (mae, mse)

    def train(self):
        """
        Run the training process for the specified number of epochs.
        Perform validation with the specified frequency.
        Log the loss components values and the validation error values
        (all that can be visualized by tensorboard).
        Save the checkpoint each time the validation is performed.
        """
        batch_iter = 0
        epoch = 0

        if self.cfg.validation.if_val_before_begin_train:
            initial_val_mae, initial_val_mse = self.validate(
                self.val_loader, step=epoch)
            initial_tr_mae, initial_tr_mse = self.validate(
                self.train_loader, step=epoch)
            self.tbx_wrtr.add_scalars(
                'error_values/MAE',
                {'val': initial_val_mae, 'train': initial_tr_mae},
                epoch)
            self.tbx_wrtr.add_scalars(
                'error_values/MSE',
                {'val': initial_val_mse, 'train': initial_tr_mse},
                epoch)

        print()
        print(f"  The checkpoints and tensorboard events files are saved to "
              f"'{os.path.relpath(self.tbx_wrtr_dir, self.orig_cwd)}'")
        print("  Progress bar for training epochs:")
        end_epoch = self.cfg.train.start_epoch + self.cfg.train.num_epochs
        for epoch in tqdm(range(self.cfg.train.start_epoch, end_epoch)):
            self.model.train()
            self.adjust_learning_rate(epoch)

            avg_lr = 0
            for param_group in self.optimizer.param_groups:
                avg_lr += param_group['lr']
            avg_lr /= len(self.optimizer.param_groups)
            self.tbx_wrtr.add_scalar(
                'hyperparams/log10_learning_rate', math.log10(avg_lr), epoch)

            for sample in self.train_loader:
                gt_cls0_label, gt_cls1_label, gt_cls2_label = sample['labels_gt']
                sample_counts_gt = [torch.unsqueeze(c, 1) for c in sample['counts_gt']]
                gt_div2 = sample_counts_gt[-1]
                image = sample['image'].float().to(self.add_cfg['device'])
                cls0_logits, cls1_logits, cls2_logits, DIV2, U1, U2, W1, W2 \
                    = self.model(image)
                self.optimizer.zero_grad()

                cross_entropy_loss_terms = loss.counter_loss(
                    (gt_cls0_label, gt_cls1_label, gt_cls2_label),
                    (cls0_logits, cls1_logits, cls2_logits))
                
                merging_loss = loss.merging_loss(gt_div2, DIV2)
                
                losses_list = [cross_entropy_loss_terms, merging_loss]
                if self.cfg.model.supervised:
                    upsampling_loss = loss.upsampling_loss(sample_counts_gt, U1, U2)
                    losses_list.append(upsampling_loss)
                    Cmax = self.model.label2count_tensor[-1]
                    division_loss = loss.division_loss(sample_counts_gt, W1, W2, Cmax)
                    losses_list.append(division_loss)
                
                total_loss = loss.total_loss(losses_list)
                
                for i, CE_term in enumerate(cross_entropy_loss_terms):
                    self.tbx_wrtr.add_scalar(
                        f'losses/cls_{i}_loss', CE_term, batch_iter)
                self.tbx_wrtr.add_scalar(
                    'losses/merging_loss', merging_loss, batch_iter)

                if self.cfg.model.supervised:
                    self.tbx_wrtr.add_scalar(
                        'losses/upsampling_loss', upsampling_loss, batch_iter)
                    self.tbx_wrtr.add_scalar(
                        'losses/division_loss', division_loss, batch_iter)
                
                self.tbx_wrtr.add_scalar(
                    'losses/total_loss', total_loss, batch_iter)

                total_loss.backward()
                self.optimizer.step()
                batch_iter += 1

            if epoch % self.cfg.validation.validate_ckpt_every_epochs == 0:
                val_mae, val_mse = self.validate(self.val_loader, step=epoch)
                tr_mae, tr_mse = self.validate(self.train_loader, step=epoch)
                self.tbx_wrtr.add_scalars(
                    'error_values/MAE',
                    {'val': val_mae, 'train': tr_mae},
                    epoch)
                self.tbx_wrtr.add_scalars(
                    'error_values/MSE',
                    {'val': val_mse, 'train': tr_mse},
                    epoch)
            
            if epoch % self.cfg.validation.save_ckpt_every_epochs == 0:
                nm = f'epoch_{epoch:04d}.pth'
                if not os.path.isdir(pjn(self.tbx_wrtr_dir, 'checkpoints')):
                    os.mkdir(pjn(self.tbx_wrtr_dir, 'checkpoints'))
                self.save_ckpt(
                    epoch=epoch,
                    fpath=pjn(self.tbx_wrtr_dir, 'checkpoints', nm))

    def adjust_learning_rate(self, epoch):
        """
        Use exponentially decreasing learning rate schedule.
        """
        lr_section = self.cfg.train.lr_schedule
        lr = lr_section.lr_init * (lr_section.lr_anneal_rate ** epoch)
        # update optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_ckpt(self, epoch, fpath):
        """
        Save the checkpoint containing the epoch index,
        the model state dict, the optimizer state dict.
        """
        d = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(d, fpath)


@hydra.main(config_path="conf", config_name="config_train_val_test.yaml")
def main(cfg):
    """
    Create data loaders, the model instance, optimizer and TrainManager()
    object. Run the training process.
    """
    orig_cwd = hydra.utils.get_original_cwd()
    train_loader, val_loader, test_loader = get_dataloaders(cfg, (1, 1, 0))
    interval_bounds, label2count_list = make_label2count_list(cfg)
    
    model = SDCNet(
        label2count_list,
        cfg.model.supervised,
        load_pretr_weights_vgg=True)

    if cfg.train.pretrained_ckpt:
        print(f"  Using pretrained model and its checkpoint "
              f"'{cfg.train.pretrained_ckpt}'")
        loaded_struct = torch.load(pjn(orig_cwd, cfg.train.pretrained_ckpt))
        model.load_state_dict(loaded_struct['model_state_dict'], strict=True)
    
    additional_cfg = {'device': None}
    if not cfg.resources.disable_cuda and torch.cuda.is_available():
        additional_cfg['device'] = torch.device('cuda')
        model = model.cuda()
    else:
        additional_cfg['device'] = torch.device('cpu')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.lr_schedule.lr_init,
        momentum=cfg.train.optimizer.momentum,
        weight_decay=cfg.train.optimizer.weight_decay)

    additional_cfg['tbx_wrtr_dir'] = os.getcwd()
    with SummaryWriter(additional_cfg['tbx_wrtr_dir']) as tbx_wrtr:
        additional_cfg['tbx_wrtr'] = tbx_wrtr
        trainer = TrainManager(
            model,
            optimizer,
            cfg,
            additional_cfg,
            train_loader=train_loader,
            val_loader=val_loader)
        trainer.train()


if __name__ == "__main__":
    main()
