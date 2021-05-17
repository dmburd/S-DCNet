import sys
import os
import os.path
from sys import exit as e
from os.path import join as pjn
import copy
import glob
import math
import random
import numpy as np
import skimage
import skimage.io
from PIL import Image
import matplotlib.pyplot as plt

import hydra
import q

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from labels_counts_utils import apply_count2label

# plt.ion()
# ^ interactive mode


def calc_rgb_mean_train(cfg):
    """
    Calculate the mean pixel value for each color channel
    for all images in the train set.
    The mean values for ShanghaiTech are typically in range 
    90--115 (approximately).

    Args:
        cfg: the global configuration (hydra).

    Returns:
        ndarray of shape (3,) containing the per-channel mean pixel values.
    """
    common_dir = pjn(
        hydra.utils.get_original_cwd(),
        cfg.dataset.dataset_rootdir,
        f'part_{cfg.dataset.part}',
        'train_data',
    )
    imgs_dir = pjn(common_dir, 'images')
    assert os.path.isdir(imgs_dir)
    jpg_files = sorted(glob.glob(pjn(imgs_dir, "*.jpg")))

    rgb_sum_vals = np.zeros((3,))
    num_pixels = 0.0

    for img_fpath in jpg_files:
        img_pil = Image.open(img_fpath)
        img_np = np.array(img_pil)
        h, w = img_np.shape[0:2]
        assert (w, h) == img_pil.size
        rgb_sum_vals += np.sum(img_np, axis=(0, 1))
        num_pixels += h * w

    return rgb_sum_vals / num_pixels


class ShanghaiTechDataset(Dataset):
    """
    The class implementing the pytorch's Dataset API.
    """

    def __init__(self, cfg, subdir, shuffle_seed=None, rel_inds=(0.0, 1.0), transform=None):
        """
        Initialization of the ShanghaiTechDataset class instance.

        Args:
            cfg: the global configuration (hydra).
            subdir: The subdirectory of the dir 'ShanghaiTech/part_B'
                or 'ShanghaiTech/part_B' where the groundtruth data is stored.
                `subdir` must be 'train_data' or 'test_data'.
            shuffle_seed: The seed for a random generator that will shuffle
                the samples in the dataset. Creating two instances of 
                ShanghaiTechDataset() with the same value of `shuffle_seed`
                for the same `subdir` will yield the same permutation of
                the samples. If `shuffle_seed` is None, no shuffling 
                is performed.
            rel_inds: Relative indices for the images to be included in the 
                dataset. If there are N images in the `subdir`, images 
                (after an optional shuffling) with indices in 
                range(int(rel_inds[0] * N), int(rel_inds[1] * N)) will be
                included in the dataset. By default rel_inds=(0.0, 1.0)
                (the whole dataset is included).
            transform: Transformations to apply to the samples 
                (object of class torchvision.transforms, for example).

        Returns:
            None.
        """
        self.transform = transform
        assert subdir in ('train_data', 'test_data')

        orig_cwd = hydra.utils.get_original_cwd()
        common_dir = pjn(
            orig_cwd,
            cfg.dataset.dataset_rootdir,
            f'part_{cfg.dataset.part}',
            subdir)

        self.imgs_dir = pjn(common_dir, 'images')

        npz_pattern = pjn(orig_cwd, cfg.dataset.densmaps_gt_npz)
        npz_files = sorted(glob.glob(npz_pattern))
        train_or_test_suffixes = \
            [os.path.splitext(os.path.split(f)[1])[0].split('_')[-1].lower()
                for f in npz_files]
        assert train_or_test_suffixes == ['test', 'train']
        npz_file = npz_files[1 if subdir == 'train_data' else 0]

        print(f"  '{os.path.relpath(npz_file, orig_cwd)}' is taken as the "
              f"density maps ground truth file for {subdir} subdirectory",
              flush=True)

        dmaps_dict = np.load(npz_file)
        img_bnames = list({k.split('/')[0] for k in dmaps_dict.keys()})
        tot_num_samples = len(img_bnames)
        i_init = int(rel_inds[0] * tot_num_samples)
        i_fin = int(rel_inds[1] * tot_num_samples)

        print(f"  Images with indices in range({i_init}, {i_fin}) will be "
              f"selected (total num. images {tot_num_samples})",
              flush=True)

        annot_bnames = sorted(
            img_bnames,
            key=lambda nm: int(nm[7:])
        )
        if shuffle_seed:
            random.Random(shuffle_seed).shuffle(annot_bnames)

        annot_bnames_selected = annot_bnames[i_init:i_fin]

        self.img_bname_list = []
        self.img_np_list = []
        self.dmap_list = []
        self.num_annot_headpoints_list = []

        print(f"  Filling the Dataset object (reading the images)... ",
              end='',
              flush=True)
        for annot_bname in annot_bnames_selected:
            self.img_bname_list.append(annot_bname[3:])
            #
            dmap = dmaps_dict[f"{annot_bname}/density_map"]
            self.dmap_list.append(dmap.astype(np.float32))
            #
            num_annot_headpoints = dmaps_dict[f"{annot_bname}/num_annot_headpoints"]
            self.num_annot_headpoints_list.append(num_annot_headpoints.astype(np.float32))
            #
            img_fpath = pjn(self.imgs_dir, annot_bname[3:] + '.jpg')
            img_np = np.array(skimage.io.imread(img_fpath))
            h, w = img_np.shape[:2]
            assert (h, w) == dmap.shape
            assert img_np.ndim == 2 or img_np.shape[2] == 3
            if img_np.ndim == 2:
                # monochrome image, only one channel; expand to 3 channels
                img_np = np.repeat(np.expand_dims(img_np, axis=2), 3, axis=2)
            self.img_np_list.append(img_np)
        
        print(f"Done", flush=True)

    def __len__(self):
        """
        Returns number of samples in the dataset.
        """
        return len(self.img_np_list)

    def __getitem__(self, idx):
        """
        Returns a sample as a dictionary containing image basename,
        image itself (as a numpy ndarray) and the corresponding
        density map. Applies transforms (if specified during
        instance initialization) before returning.
        """
        sample = {
            'image_bname': self.img_bname_list[idx],
            # ^ str like 'IMG_2'
            #   (batching: several strings are packed to a list)
            'image': self.img_np_list[idx],
            # ^ shape (h, w, 3) (for example (768, 1024, 3))
            #   values 0..255
            #   (batching: leading dimension 1 added)
            'dmap': self.dmap_list[idx],
            # ^ shape (h, w) (for example (768, 1024))
            #   (batching: leading dimension 1 added)
            'num_annot_headpoints': self.num_annot_headpoints_list[idx],
            # ^ shape () (a single scalar, the total number of annotated heads)
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize(object):
    """
    Apply this transform first!
    Subtracts per-channel mean pixel values from the image pixels
    and divides the difference by 255.
    """

    def __init__(self, rgb_mean_vals):
        self.rgb_mean_vals = rgb_mean_vals

    def __call__(self, sample):
        normd_image = (sample['image'] - self.rgb_mean_vals) / 255
        return {
            'image_bname': sample['image_bname'],
            'image': normd_image.astype(np.float32),
            # ^ normalized values in a range like [-0.5; +0.5]
            #   shape (h, w, 3)
            'dmap': sample['dmap'],
            'num_annot_headpoints': sample['num_annot_headpoints'],
        }


class ToCHW(object):
    """
    Apply this transform second, right after Normalize()!
    Move the color channels axis to the 1st position because the order is
    HWC after loading an image to numpy ndarray, but CHW for pytorch.
    """

    def __call__(self, sample):
        return {
            'image_bname': sample['image_bname'],
            'image': sample['image'].transpose((2, 0, 1)),
            # ^ shape (3, h, w)
            'dmap': sample['dmap'],
            # ^ shape (h, w)
            'num_annot_headpoints': sample['num_annot_headpoints'],
        }


# see torchvision/transforms/transforms.py, class RandomHorizontalFlip
class RandomHorizontalFlip(object):
    """Horizontally flip the given torch tensor randomly with a given probability.
    Must be applied after ToCHW() because CHW order is assumed here.
    The dimension corresponding to width is passed to np.flip().
    Without .copy(), the result would have negative stride. It is not supported.
    """

    def __init__(self, p=0.5):
        """
        Args:
            p (float): probability of the image being flipped. Default: 0.5
        """
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return {
                'image_bname': sample['image_bname'],
                'image': np.flip(sample['image'], 2).copy(),
                # 2 (the 2nd arg to np.flip()) is the width dimension here ^
                'dmap': np.flip(sample['dmap'], 1).copy(),
                # 1 (the 2nd arg to np.flip()) is the width dimension here ^
                'num_annot_headpoints': sample['num_annot_headpoints'],
            }
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class QuasiRandomCrop(object):
    """
    Augmentation used in CSRNet (https://arxiv.org/abs/1802.10062).
    9 sub-images of 1/4 resolution are cropped from the original image.
    The first 4 sub-images are from four corners, 
    and the remaining 5 are randomly cropped.
    """

    def __call__(self, sample):
        h, w = sample['image'].shape[1], sample['image'].shape[2]
        assert (h, w) == sample['dmap'].shape
        h_half, w_half = int(h/2), int(w/2)

        r = random.random()
        if r <= 1/9:
            h1, h2 = 0, h_half
            w1, w2 = 0, w_half
            bname_add = "_upper_left_crop"
        elif 1/9 < r <= 2/9:
            h1, h2 = 0, h_half
            w1, w2 = w_half, w
            bname_add = "_upper_right_crop"
        elif 2/9 < r <= 3/9:
            h1, h2 = h_half, h
            w1, w2 = 0, w_half
            bname_add = "_lower_left_crop"
        elif 3/9 < r <= 4/9:
            h1, h2 = h_half, h
            w1, w2 = w_half, w
            bname_add = "_lower_right_crop"
        else:
            h1 = int(random.random() * h_half)
            h2 = h1 + h_half
            w1 = int(random.random() * w_half)
            w2 = w1 + w_half
            bname_add = "_random_crop"

        return {
            'image_bname': sample['image_bname'] + bname_add,
            'image': sample['image'][:, h1:h2, w1:w2],
            # ^ shape (3, h_half, w_half)
            'dmap': sample['dmap'][h1:h2, w1:w2],
            # ^ shape (h_half, w_half)
            'num_annot_headpoints': sample['num_annot_headpoints'],
        }


class PadToMultipleOf64(object):
    """
    Pad the image (and the density map) to a size of (64*k, 64*n)
    where k and n are integers.
    """

    def __call__(self, sample):
        h, w = sample['image'].shape[1], sample['image'].shape[2]
        h_pd = math.ceil(h / 64) * 64
        h_pd_diff = h_pd - h
        w_pd = math.ceil(w / 64) * 64
        w_pd_diff = w_pd - w
        pad_pairs = (
            (h_pd_diff // 2, h_pd_diff - h_pd_diff // 2),
            # ^ (above the upper edge, below the lower edge)
            (w_pd_diff // 2, w_pd_diff - w_pd_diff // 2),
            # ^ (to the left side of the left edge, to the right of the right)
        )

        image_pd = np.pad(
            sample['image'],
            pad_width=((0, 0), pad_pairs[0], pad_pairs[1]),
            mode='constant',
            constant_values=0)

        dmap_pd = np.pad(
            sample['dmap'],
            pad_width=(pad_pairs[0], pad_pairs[1]),
            mode='constant',
            constant_values=0.0)

        return {
            'image_bname': sample['image_bname'],
            'image': image_pd,
            # ^ shape (3, h_pd, w_pd)
            'dmap': dmap_pd,
            # ^ shape (h_pd, w_pd)
            'num_annot_headpoints': sample['num_annot_headpoints'],
        }


class AddGtCountsLabels(object):
    """
    Calculate `labels_gt` and `div2_gt`
    and add the values to the sample dict.
    """

    def __init__(self, interval_bounds):
        self.interval_bounds = interval_bounds

    def __call__(self, sample):
        """
        Perform calculations for the density map. 
        Tensors (`labels_gt`) containing class labels for patches of sizes
        64x64, 32x32 and 16x16 are retrieved based on the ground truth
        count values (`counts_gt`) for the corresponding patches.
        Also, the ground truth count values (`div2_gt`) for the smallest patches
        (16x16) are kept separately.
        Two new key-value pairs (corresponding to `counts_gt` and `div2_gt`)
        are added to the `sample`'s dictionary.

        The ground truth count values are obtained by applying a conv2d
        operation with trivial kernels 64x64, 32x32 and 16x16 that 
        contain ones.
        """
        inp = torch.from_numpy(sample['dmap'].astype(np.float32))
        inp.unsqueeze_(0).unsqueeze_(0)

        counts_gt = []
        for patch_size in [64, 32, 16]:
            p = patch_size
            kernel = torch.ones((1, 1, p, p))
            conv_out = F.conv2d(input=inp, weight=kernel, stride=p)
            counts_gt.append(conv_out.squeeze_(0).squeeze_(0))

        labels_gt = [
            apply_count2label(count_gt, self.interval_bounds)
            for count_gt in counts_gt]

        return {
            'image_bname': sample['image_bname'],
            'image': sample['image'],
            'dmap': sample['dmap'],
            'num_annot_headpoints': sample['num_annot_headpoints'],
            'counts_gt': counts_gt,
            'labels_gt': labels_gt,
        }


class NoModifications(object):
    """
    Identity transform of a sample.
    Does nothing. Was used for debugging.
    """

    def __call__(self, sample):
        print("\tNoModifications() called!", flush=True)
        for attr in ['image', 'dmap', 'div2_gt']:
            print(f"sample['{attr}']:", type(sample[attr]),
                  sample[attr].dtype, sample[attr].shape)
        for ndarr in sample['labels_gt']:
            print("  one of sample['labels_gt']:",
                  type(ndarr), ndarr.dtype, ndarr.shape)

        return sample
