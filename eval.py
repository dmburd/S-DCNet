#!/usr/bin/python3

import sys
from sys import exit as e
import os
import os.path
from os.path import join as pjn
import q
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from labels_counts_utils import make_label2count_list
import ShanghaiTech_dataset as dtst
from SDCNet import SDCNet
import train


def update_parser_for_arguments(parser_from_train):
    """
    Add one argument to the passed ArgumentParser() object 
    and return the modified object.
    """
    parser_from_train.add_argument(
        '--checkpoint', metavar='CKPT_PATH',
        type=str,
        default=None,
        help="path (required) to the checkpoint to be evaluated"
             "(default: None)")

    return parser_from_train


def visualize_predictions(model, data_loader, vis_dir):
    """
    Visualize model predictions by running inference on the samples and
    create (and save to disk) compound images that contain the original
    image, the ground truth count, the predicted count and the approximate
    predicted density map represented by the tensor `DIV2` (local count values
    for 16x16 patches).

    Args:
        model: the model to be run on the provided samples.
        data_loader: DataLoader object that provides samples.
        vis_dir: path to the directory where the compound images will be 
            placed.

    Returns: 
        None.

    Side effects:
        Creates compound images (for visualization purposes) in `vis_dir`.
    """
    if not os.path.isdir(vis_dir):
        os.mkdir(vis_dir)

    model.eval()
    rgb_mean_train = dtst.calc_rgb_mean_train(args_dict)
    rgb_mean_train = rgb_mean_train[:, np.newaxis, np.newaxis]

    for sample in data_loader:
        gt_count = sample['dmap'].numpy().sum()
        image_normd = sample['image'].float().to('cuda')
        cls0_logits, cls1_logits, cls2_logits, DIV2 = model(image_normd)
        pred_count_16x16_blocks = DIV2.cpu().detach().numpy()
        pred = pred_count_16x16_blocks.squeeze(0).squeeze(0)
        pred_count = pred.sum()
        if gt_count >= 1:
            rel_err_percent = (pred_count - gt_count) / gt_count * 100
        else:
            rel_err_percent = 0.0
        image = image_normd.cpu().numpy() * 255 + rgb_mean_train
        image = image.astype(int).squeeze(0)
        # ^ shape = (3, w, h)
        image = np.transpose(image, (1, 2, 0))
        bleach_coef = 0.4
        image_bleached = (255 - (255 - image) * bleach_coef).astype(int)
        bname = sample['image_bname'][0]
        # ^ bnames appear in numerically sorted order ('IMG_1', 'IMG_2', ...)

        #h, w = image.shape[:2]
        w, h = 1024, 768
        # search for "Relationship between dpi and figure size"
        # on stackoverflow
        dpi = 100
        # ^ dpi = 100 by default
        fs_h = int(h / dpi)
        fs_w = int(w / dpi)
        fs = (fs_w, fs_h)
        fig, axs = plt.subplots(figsize=fs, dpi=dpi, ncols=2)
        axs[0].imshow(image, vmin=0, vmax=255)
        axs[0].set_title('Ground truth total count = %.1f' % gt_count)
        extent = (0, w, 0, h)
        axs[1].imshow(image_bleached, vmin=0, vmax=255, extent=extent)
        pred_im = axs[1].imshow(
            pred,
            cmap='bwr', alpha=0.5,
            extent=extent, interpolation='nearest')
        axs[1].set_title(
            'Predicted total count = %.1f\n(error = %+.1f, '
            'relative error = %+.1f%%)'
            % (pred_count, pred_count - gt_count, rel_err_percent))
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pred_im, cax=cax)
        fig.tight_layout()
        plt.savefig(pjn(vis_dir, bname + '.png'), bbox_inches='tight')
        plt.close(fig)


def main(args_dict):
    print("  Evaluating the checkpoint '%s'" % args_dict['checkpoint'])
    loaded_struct = torch.load(args_dict['checkpoint'])

    args_dict.update(loaded_struct['args_dict_copy'])
    args_dict.update(train_val_split=1.0)
    # ^ place all of the train data to the train_loader below
    #   (do not split the train data into train + validation)
    train_loader, val_loader, test_loader = \
        train.get_dataloaders(args_dict, (1, 0, 1))

    interval_bounds, label2count_list = make_label2count_list(args_dict)
    model = SDCNet(label2count_list, load_pretr_weights_vgg=False)
    model.load_state_dict(loaded_struct['model_state_dict'], strict=True)
    model = model.cuda()

    optimizer = None

    trainer = train.TrainManager(
        model,
        optimizer,
        args_dict,
        train_loader=None,
        val_loader=None,
    )

    print()
    datadir = pjn(args_dict['dataset_rootdir'], 'part_%s' % args_dict['part'])
    print("  Evaluating on the (whole) train data and on the test data "
          "(in %s)" % datadir)

    mae_train, mse_train = trainer.validate(train_loader)
    print("  Metrics on the (whole) train data: MAE: %.2f, MSE: %.2f"
          % (mae_train, mse_train))

    mae_test, mse_test = trainer.validate(test_loader)
    print("  Metrics on the test data:          MAE: %.2f, MSE: %.2f"
          % (mae_test, mse_test))

    visualize_predictions(
        model,
        test_loader,
        'visualized_part_%s_predictions' % args_dict['part'])


if __name__ == "__main__":
    """
    `parser_for_arguments()` from 'train.py' is used.
    It means that this script supports all command line arguments that
    'train.py' supports (plus '--checkpoint' argument added by the function
    `update_parser_for_arguments()` in this file).

    Typically, all parameters required for restoring the model are obtained
    from 'args_dict_copy' field of the struct loaded from the provided
    checkpoint.
    """
    parser_from_train = train.parser_for_arguments()
    update_parser_for_arguments(parser_from_train)
    args = parser_from_train.parse_args()

    args_dict = vars(args)
    main(args_dict)
