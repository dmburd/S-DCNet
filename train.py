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
import argparse
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


def str2bool(v):
    """
    Interprets any of the several specified string values
    as `True` and all other values as `False`.
    Used for command line argument parsing.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False


def parser_for_arguments():
    """
    Creates an ArgumentParser() object, calls its method `.add_argument()`
    a number of times and returns the object.
    Note: `.parse_args()` is not called inside this function.
    Another function can be applied in order to call `.add_argument()`
    again later.

    Args:
        None.

    Returns:
        argparse.ArgumentParser() object.
    """
    parser = argparse.ArgumentParser(description='S-DCNet training script')

    parser.add_argument(
        '--disable-cuda', metavar='{yes|no}',
        type=str2bool,
        default=False,
        help="'yes' to disable CUDA (default: no, CUDA enabled)")

    parser.add_argument(
        '--supervised', metavar='{yes|no}',
        type=str2bool,
        default=False,
        help="'yes' for the Supervised S-DCNet (SS-DCNet), "
             "'no' for the older version (unsupervised, ordinary S-DCNet) "
             "(default: no)")

    parser.add_argument(
        '--dataset-rootdir', metavar='D',
        type=str,
        default='./ShanghaiTech',
        help="root dir for the dataset (default: './ShanghaiTech')")

    parser.add_argument(
        '--part', metavar='A_or_B',
        type=str,
        default='B',
        help="'A' for part_A, 'B' for part_B (default: 'B')")

    parser.add_argument(
        '--densmaps-gt-npz', metavar='P',
        type=str,
        default='./density_maps_part_B_*.npz',
        help="shell pattern for the path to the files "
             "'density_maps_part_{A|B}_{train,test}.npz' "
             "(default: './density_maps_part_B_*.npz')")

    parser.add_argument(
        '--train-val-split', metavar='S',
        type=float,
        default=0.9,
        help="fraction (S) of the training set is used for training, "
             "fraction (1-S) of the training set is used for validation "
             "(default: 0.9)")

    parser.add_argument(
        '--train-batch-size', metavar='N',
        type=int,
        default=1,
        help="batch size for training (default: 1)")

    parser.add_argument(
        '--val-batch-size', metavar='N',
        type=int,
        default=1,
        help="batch size for validation (default: 1)")

    parser.add_argument(
        '--test-batch-size', metavar='N',
        type=int,
        default=1,
        help="batch size for testing (default: 1)")

    parser.add_argument(
        '--pretrained-model', metavar='PRTRN_PATH',
        type=str,
        default=None,
        help="start training from this pretrained model (path is required)"
             "(default: None, no pretraining, start from scratch)")

    parser.add_argument(
        '--lr', metavar='INIT_LR',
        type=float,
        default=1e-3,
        help="initial learning rate for the optimizer (default: 1e-3)")

    parser.add_argument(
        '--lr-anneal-rate', metavar='A',
        type=float,
        default=0.99,
        help="lr annealing rate (lr = INIT_LR * A**epoch) (default: 0.99)")

    parser.add_argument(
        '--momentum', metavar='M',
        type=float,
        default=0.9,
        help="momentum for the optimizer (default: 0.9)")

    parser.add_argument(
        '--weight-decay', metavar='D',
        type=float,
        default=1e-4,
        help="weight decay for the optimizer (default: 1e-4)")

    parser.add_argument(
        '--num-epochs', metavar='NE',
        type=int,
        default=1000,
        help="number of epochs for training (default: 1000)")

    parser.add_argument(
        '--start-epoch', metavar='SE',
        type=int,
        default=1,
        help="manually set idx for the starting epoch "
             "(useful on restarts) (default: 1)")

    parser.add_argument(
        '--validate-every-epochs', metavar='VE',
        type=int,
        default=10,
        help="validate every VE epochs (default: 10)")

    parser.add_argument(
        '--num-intervals', metavar='N',
        type=int,
        default=7,
        help="number of intervals for count values "
             "(set 22 for 'part_A', 7 for 'part_B') (default: 7)")

    parser.add_argument(
        '--interval-step', metavar='S',
        type=float,
        default=0.5,
        help="interval size for count values (default: 0.5)")

    parser.add_argument(
        '--partition-method', metavar='M',
        type=int,
        default=2,
        help="partition method (1 for one-linear, 2 for two-linear) "
             "(default: 2)")

    parser.add_argument(
        '--verbose', metavar='{yes|no}',
        type=str2bool,
        default=True,
        help="print some useful info to stdout (default: yes)")

    return parser


def cur_datetime_str():
    """
    A custom pretty-printed date and time (at the moment of function call).
    """
    t = datetime.now().timetuple()
    ans = "%d-%02d-%02d_%02d-%02d-%02d" % t[:6]
    return ans


def print_dbg_info_dataloader(loader):
    """
    Prints image names and tensor shapes for samples in a DataLoader.
    Was used for debugging.
    """
    for i, sample in enumerate(loader):
        print("i = %d; image bname = %s; image shape = %s"
              % (i, sample['image_bname'], tuple(sample['image'].shape)))
        print("counts_gt shapes = %s; labels_gt shapes = %s"
              % (str(tuple(l.shape) for l in sample['labels_gt']),
                 str(tuple(l.shape) for l in sample['counts_gt'])))
        print()


def get_dataloaders(args_dict, train_val_test_mask):
    """
    Constructs Dataset objects and corresponding DataLoader objects.

    Args:
        args_dict: Dictionary containing required configuration values.
            The keys required for this function are 'interval_bounds',
            'interval_step', 'num_intervals', 'partition_method',
            'dataset_rootdir', 'part', 'train_val_split',
            'train_batch_size', 'val_batch_size', 'test_batch_size'.
        train_val_test_mask: a list or tuple containing three values
            that will be interpreted as booleans, for example (1,1,0).
            The corresponding DataLoaders for the (train, val, test)
            filtered list of Datasets will be created and returned.

    Returns:
        The tuple (train_loader, val_loader, test_loader). The loaders
        corresponding to the `False` values in `train_val_test_mask` 
        are set to `None`.
    """
    args_dict['interval_bounds'], label2count_list = \
        make_label2count_list(args_dict)

    assert len(label2count_list) \
        == args_dict['interval_bounds'].shape[0] + 1, \
        "Number of class labels must be equal to (1 + number of interval " \
        "boundaries), but the equality does not hold"

    rgb_mean_train = dtst.calc_rgb_mean_train(args_dict)
    args_dict['rgb_mean_train'] = rgb_mean_train

    composed_transf_train = transforms.Compose([
        dtst.Normalize(rgb_mean_train),
        dtst.ToCHW(),
        dtst.RandomHorizontalFlip(p=0.5),
        dtst.QuasiRandomCrop(),
        dtst.PadToMultipleOf64(),
        dtst.AddGtCountsLabels(args_dict['interval_bounds']),
    ])

    composed_transf_test = transforms.Compose([
        dtst.Normalize(rgb_mean_train),
        dtst.ToCHW(),
        dtst.PadToMultipleOf64(),
    ])

    if train_val_test_mask[0]:
        train_dataset = dtst.ShanghaiTechDataset(
            args_dict,
            subdir='train_data',
            shuffle_seed=42,
            rel_inds=(0.0, args_dict['train_val_split']),
            transform=composed_transf_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args_dict['train_batch_size'],
            shuffle=True,
            num_workers=4)
    else:
        train_loader = None

    if train_val_test_mask[1]:
        val_dataset = dtst.ShanghaiTechDataset(
            args_dict,
            subdir='train_data',
            shuffle_seed=42,
            rel_inds=(args_dict['train_val_split'], 1.0),
            transform=composed_transf_test)
        val_loader = DataLoader(
            train_dataset,
            batch_size=args_dict['val_batch_size'],
            shuffle=False,
            num_workers=4)
    else:
        val_loader = None

    if train_val_test_mask[2]:
        test_dataset = dtst.ShanghaiTechDataset(
            args_dict,
            subdir='test_data',
            shuffle_seed=None,
            rel_inds=(0.0, 1.0),
            transform=composed_transf_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args_dict['test_batch_size'],
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
            args_dict,
            train_loader,
            val_loader,
            if_val_before_begin_train=False):
        """
        Save a number of configuration parameters as instance attributes.
        `tbx_wrtr` and `tbx_wrtr_dir` should not be serialized, so they
        are excluded from the copy of the configuration `args_dict`.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = args_dict['device']
        self.supervised = args_dict['supervised']
        self.num_epochs = args_dict['num_epochs']
        self.validate_every_epochs = args_dict['validate_every_epochs']
        self.if_val_before_begin_train = if_val_before_begin_train
        self.start_epoch = args_dict['start_epoch']
        self.init_learning_rate = args_dict['lr']
        self.lr_anneal_rate = args_dict['lr_anneal_rate']
        self.tbx_wrtr_dir = args_dict['tbx_wrtr_dir']
        self.tbx_wrtr = args_dict['tbx_wrtr']
        self.verbose = args_dict['verbose']
        self.args_dict_copy = {
            k: v for k, v in args_dict.items()
            if k not in ('device', 'tbx_wrtr', 'tbx_wrtr_dir')
        }
        self.args_dict_copy.update(
            device=None,
            tbx_wrtr=None,
            tbx_wrtr_dir=None)

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
                gt_count = sample['dmap'].numpy().sum()
                image = sample['image'].float().to(self.device)
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

        if self.if_val_before_begin_train:
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
        print("  The checkpoints and tensorboard events files "
              "are saved to '%s'" % self.tbx_wrtr_dir)
        print("  Progress bar for training epochs:")
        end_epoch = self.start_epoch + self.num_epochs
        for epoch in tqdm(range(self.start_epoch, end_epoch)):
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
                image = sample['image'].float().to(self.device)
                cls0_logits, cls1_logits, cls2_logits, DIV2, U1, U2, W1, W2 \
                    = self.model(image)
                self.optimizer.zero_grad()

                cross_entropy_loss_terms = loss.counter_loss(
                    (gt_cls0_label, gt_cls1_label, gt_cls2_label),
                    (cls0_logits, cls1_logits, cls2_logits))
                
                merging_loss = loss.merging_loss(gt_div2, DIV2)
                
                losses_list = [cross_entropy_loss_terms, merging_loss]
                if self.supervised:
                    upsampling_loss = loss.upsampling_loss(sample_counts_gt, U1, U2)
                    losses_list.append(upsampling_loss)
                    Cmax = self.model.label2count_tensor[-1]
                    division_loss = loss.division_loss(sample_counts_gt, W1, W2, Cmax)
                    losses_list.append(division_loss)
                
                total_loss = loss.total_loss(losses_list)
                
                for i, CE_term in enumerate(cross_entropy_loss_terms):
                    self.tbx_wrtr.add_scalar(
                        'losses/cls_%d_loss' % i, CE_term, batch_iter)
                self.tbx_wrtr.add_scalar(
                    'losses/merging_loss', merging_loss, batch_iter)

                if self.supervised:
                    self.tbx_wrtr.add_scalar(
                        'losses/upsampling_loss', upsampling_loss, batch_iter)
                    self.tbx_wrtr.add_scalar(
                        'losses/division_loss', division_loss, batch_iter)
                
                self.tbx_wrtr.add_scalar(
                    'losses/total_loss', total_loss, batch_iter)

                total_loss.backward()
                self.optimizer.step()
                batch_iter += 1

            if epoch % self.validate_every_epochs == 0:
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
                #
                nm = 'epoch_%04d.pth' % epoch
                if not os.path.isdir(pjn(self.tbx_wrtr_dir, 'checkpoints')):
                    os.mkdir(pjn(self.tbx_wrtr_dir, 'checkpoints'))
                self.save_ckpt(epoch, fpath=pjn(
                    self.tbx_wrtr_dir, 'checkpoints', nm))

    def adjust_learning_rate(self, epoch):
        """
        Use exponentially decreasing learning rate schedule.
        """
        lr = self.init_learning_rate * (self.lr_anneal_rate ** epoch)
        # update optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_ckpt(self, epoch, fpath):
        """
        Save the checkpoint containing the epoch index,
        the model state dict, the optimizer state dict,
        and the configuration parameters (`args_dict`).
        """
        d = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args_dict_copy': self.args_dict_copy,
        }
        torch.save(d, fpath)


def main(args_dict):
    """
    Create data loaders, the model instance, optimizer and TrainManager()
    object. Run the training process.
    """
    train_loader, val_loader, test_loader = get_dataloaders(
        args_dict, (1, 1, 0))

    interval_bounds, label2count_list = make_label2count_list(args_dict)
    
    model = SDCNet(
        label2count_list,
        args_dict['supervised'],
        load_pretr_weights_vgg=True)

    if args_dict['pretrained_model']:
        print("  Using pretrained model and its checkpoint '%s'"
              % args_dict['pretrained_model'])
        loaded_struct = torch.load(args_dict['pretrained_model'])
        model.load_state_dict(loaded_struct['model_state_dict'], strict=True)

    args_dict['device'] = None
    if not args_dict['disable_cuda'] and torch.cuda.is_available():
        args_dict['device'] = torch.device('cuda')
        model = model.cuda()
    else:
        args_dict['device'] = torch.device('cpu')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args_dict['lr'],
        momentum=args_dict['momentum'],
        weight_decay=args_dict['weight_decay'])

    trainer = TrainManager(
        model,
        optimizer,
        args_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        if_val_before_begin_train=False)

    trainer.train()


if __name__ == "__main__":
    parser = parser_for_arguments()
    args = parser.parse_args()

    tbx_wrtr_dir = "run_%s" % cur_datetime_str()

    with SummaryWriter(tbx_wrtr_dir) as tbx_wrtr:
        args_dict = vars(args)
        args_dict['tbx_wrtr_dir'] = tbx_wrtr_dir
        args_dict['tbx_wrtr'] = tbx_wrtr
        main(args_dict)
