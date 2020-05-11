#!/usr/bin/python3

import sys
from sys import exit as e
import os
import os.path
from os.path import join as pjn
import q
import skimage
import skimage.io
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
    Add a few arguments to the passed ArgumentParser() object 
    and return the modified object.
    """
    parser_from_train.add_argument(
        '--checkpoint', metavar='CKPT_PATH',
        type=str,
        default=None,
        help="path (required) to the checkpoint to be evaluated"
             "(default: None)")

    parser_from_train.add_argument(
        '--images-dir', metavar='D',
        type=str,
        default=None,
        help="path (required) to the directory containing images"
             "(default: None)")

    parser_from_train.add_argument(
        '--export', metavar='{yes|no}',
        type=train.str2bool,
        default=False,
        help="export the model in several formats (default: no)")

    return parser_from_train


def visualize_predictions(model, args_dict):
    """
    Visualize model predictions by running inference on the samples and
    create (and save to disk) compound images that contain the original
    image, the predicted count and the approximate predicted density map
    represented by the tensor `DIV2` (local count values for 16x16 patches).
    """
    model.eval()
    rgb_mean_train = args_dict['rgb_mean_train']

    img_dir = args_dict['images_dir']
    vis_dir = pjn(img_dir, 'visualized_predictions')

    flist = sorted(os.listdir(img_dir))
    fpaths = [pjn(img_dir, fname) for fname in flist]
    fpaths = [fpath for fpath in fpaths if os.path.isfile(fpath)]
    if fpaths:
        print('  <image_filename>: <total_count>')
        if not os.path.isdir(vis_dir):
            os.mkdir(vis_dir)

    for fpath in fpaths:
        fname = os.path.split(fpath)[1]
        try:
            img_np = np.array(skimage.io.imread(fpath))
        except:
            if os.path.isfile(fpath):
                print(f"  {fname}: cannot read this file as an image")
            continue

        h, w = img_np.shape[:2]
        assert img_np.ndim == 2 or img_np.shape[2] == 3
        if img_np.ndim == 2:
            # monochrome image, only one channel; expand to 3 channels
            img_np = np.repeat(np.expand_dims(img_np, axis=2), 3, axis=2)

        normd_image = (img_np - rgb_mean_train) / 255
        sample = {
            'image_bname': os.path.splitext(fname)[0],
            'image': normd_image.astype(np.float32).transpose((2, 0, 1)),
            'dmap': np.zeros((h, w)),
        }
        sample_pd = dtst.PadToMultipleOf64()(sample)
        sample_pd['image'] = np.expand_dims(sample_pd['image'], axis=0)
        image_normd = torch.from_numpy(sample_pd['image']).float().to('cuda')
        cls0_logits, cls1_logits, cls2_logits, DIV2 = model(image_normd)
        pred_count_16x16_blocks = DIV2.cpu().detach().numpy()
        pred = pred_count_16x16_blocks.squeeze(0).squeeze(0)
        pred_count = pred.sum()
        print(f"  {fname}: {pred_count:.1f}")
        #
        bleach_coef = 0.4
        image_bleached = (255 - (255 - img_np) * bleach_coef).astype(int)
        bname = sample['image_bname']
        w, h = 1024, 768
        dpi = 100
        fs_h = int(h / dpi)
        fs_w = int(w / dpi)
        fs = (fs_w, fs_h)
        fig, axs = plt.subplots(figsize=fs, dpi=dpi, ncols=2)
        axs[0].imshow(img_np, vmin=0, vmax=255)
        axs[0].set_title('Original image')
        extent = (0, w, 0, h)
        axs[1].imshow(image_bleached, vmin=0, vmax=255, extent=extent)
        pred_im = axs[1].imshow(
            pred,
            cmap='bwr', alpha=0.5,
            extent=extent, interpolation='nearest')
        axs[1].set_title(f"Predicted total count = {pred_count:.1f}")
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pred_im, cax=cax)
        fig.tight_layout()
        plt.savefig(pjn(vis_dir, bname + '.png'), bbox_inches='tight')
        plt.close(fig)


def main(args_dict):
    print(f"  Running inference using checkpoint '{args_dict['checkpoint']}'")
    loaded_struct = torch.load(args_dict['checkpoint'])

    args_dict.update(loaded_struct['args_dict_copy'])

    interval_bounds, label2count_list = make_label2count_list(args_dict)
    model = SDCNet(label2count_list, load_pretr_weights_vgg=False)
    model.load_state_dict(loaded_struct['model_state_dict'], strict=True)
    model = model.cuda()

    visualize_predictions(model, args_dict)

    if args_dict['export']:
        batch_size = 1
        x = torch.randn(batch_size, 3, 64*1, 64*1, requires_grad=False).cuda()
        #
        p1, ext = os.path.splitext(args_dict['checkpoint'])
        torch.onnx.export(model, x, p1 + ".onnx", opset_version=11)
        #
        traced_script_module = torch.jit.trace(model, x)
        traced_script_module.save(p1 + "_jit_trace.pt")
        #
        script_module = torch.jit.script(model)
        script_module.save(p1 + "_jit_script.pt")


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
