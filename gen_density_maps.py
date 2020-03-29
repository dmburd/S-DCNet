#!/usr/bin/python3

import sys
import os
import os.path
import glob
from sys import exit as e
from os.path import join as pjn
import q

import math
import random
from collections import OrderedDict
import argparse
import tempfile
import time
from tqdm import tqdm

import skimage
import skimage.io
from PIL import Image
import numpy as np
import scipy.io
from sklearn.neighbors import NearestNeighbors

# skimage throws a lot of warnings like
# /usr/local/lib/python3.5/dist-packages/skimage/io/_io.py:141:
#    UserWarning: fname.png is a low contrast image.
# Let's suppress them.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_arguments():
    """
    Creates an ArgumentParser() object, calls its method `.add_argument()`
    a number of times, calls `.parse_args()` and returns the result.
    """
    parser = argparse.ArgumentParser(
        description='Density map generator for ShanghaiTech (part_A, part_B)')

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
        '--knn', metavar='K',
        type=int,
        default=3,
        help="number of nearest neigbors to "
             "calculate distance to (default: 3)")

    parser.add_argument(
        '--max-knn-avg-dist', metavar='M',
        type=float,
        default=50.0,
        help="average knn distance is set to M if exceeds it (default M: 50)")

    parser.add_argument(
        '--sigma-coef', metavar='COEF',
        type=float,
        default=0.3,
        help="gaussian's sigma = COEF * knn_avg_dist (default COEF: 0.3)")

    parser.add_argument(
        '--sqr-side', metavar='S',
        type=int,
        default=40,
        help="gaussian values set to 0.0 outside of [-S/2, +S/2] "
             "(default S: 40)")

    parser.add_argument(
        '--xhp-dir', metavar='PATH',
        type=str,
        help="path to the dir containing density maps as *.mat files "
             "(for part_A or part_B test set) from the repo "
             "github.com/xhp-hust-2018-2011/S-DCNet")

    args = parser.parse_args()
    return args


def check_consist_imgs_annots(imgs_dir, annot_dir):
    """
    Check the correspondence between the images and annotations.
    `imgs_dir` must contain '*.jpg' files, `annot_dir` must contain 
    the same number of '*.mat' files, the basenames of the files must
    differ only by the leading 'GT_' substring
    (in the '*.mat' file basenames).
    """
    assert os.path.isdir(imgs_dir), \
        "images directory '%s' is not found" % imgs_dir
    jpg_files = sorted(glob.glob(pjn(imgs_dir, "*.jpg")))
    assert jpg_files, "directory '%s' contains no '*.jpg' files" % imgs_dir

    jpg_basenames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in jpg_files]

    assert os.path.isdir(annot_dir), \
        "annotations directory '%s' is not found" % annot_dir
    mat_files = sorted(glob.glob(pjn(annot_dir, "*.mat")))
    assert mat_files, "directory '%s' contains no '*.mat' files" % annot_dir

    mat_basenames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in mat_files]

    assert len(jpg_basenames) == len(mat_basenames), \
        "different number of image files and annotation files"

    corresp_basenames = [
        (bn_mat == "GT_" + bn_jpg)
        for bn_jpg, bn_mat in zip(jpg_basenames, mat_basenames)
    ]
    assert all(corresp_basenames), \
        "image and ground truth file basenames are not consistent"


def get_headpoints_dict(annot_dir):
    """
    Load the '*.mat' files from the annotation directory
    and convert their contents (coordinates of the head points)
    to the {basename: numpy ndarray} dictionary (OrderedDict()).
    """
    mat_files = sorted(glob.glob(pjn(annot_dir, "*.mat")))
    mat_basenames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in mat_files]
    basename2headpoints_dict = OrderedDict()
    for f, bn in zip(mat_files, mat_basenames):
        mat = scipy.io.loadmat(f)
        numpy_void_obj = mat['image_info'][0][0][0][0]
        headpoints = numpy_void_obj[0]
        num_headpoints = numpy_void_obj[1][0][0]
        assert headpoints.shape[0] == num_headpoints, \
            "number of headpoints entries != specified " \
            "total number of headpoints"
        assert headpoints.shape[1] == 2, \
            "<2 or >2 coordinate values for one headpoint entry"
        basename2headpoints_dict[bn] = headpoints

    return basename2headpoints_dict


def get_one_head_gaussian(side_len, r, sigma):
    """
    Pre-calculate the values of the Gaussian function in the 
    specified spatial square region (in the points with integer
    coordinates).

    Args:
        side_len: side of the square inside which the Gaussian values
            should be calculated.
        r: the Gaussian is cenetered in the point (r, r).
        sigma: the Gaussian RMS width.

    Returns:
        Two-dimensional array containing the Gaussian function values.
    """
    one_head_gaussian = np.zeros((side_len + 2, side_len + 2))
    for i in range(side_len + 1):
        for j in range(side_len + 1):
            t = -(i - r + 1)**2 - (j - r + 1)**2
            t /= 2 * sigma**2
            one_head_gaussian[i, j] = math.exp(t) / (sigma**2 * 2*math.pi)

    return one_head_gaussian


def generate_density_maps(basename2headpoints_dict, imgs_dir, args_dict):
    """
    Generate the density maps. They are the sums of normalized Gaussian
    functions centered at the people's head points.

    Implementation details: for each headpoint, a Gaussian 2d array is 
    constructed. It is clipped to the image boundaries if needed.
    The remaining part of the array is normalized such that the values 
    corresponding to one head sum to 1. Density map is the sum of all
    (normalized) Gaussians for all heads. Total sum of the density map
    values is equal to the number of annotated heads.

    The Gaussian RMS width is adaptive. Consider one head point. 
    `args_dict['knn']` (3 by default) nearest neighbors for that point
    are found and average distance to them is calculated. That average 
    distance is capped by the constant pre-defined value 
    `args_dict['max_knn_avg_dist']` (50.0 by default) for ShanghaiTech 
    part_B dataset (the average distance is not capped for ShanghaiTech
    part_A). The `sigma` (Gaussian RMS width) is the product of the average
    distance and a pre-defined constant `args_dict['sigma_coef']`
    (0.3 by default).
    The sum of the density map values across the whole image area must be
    equal to the number of annotated heads.

    Args:
        basename2headpoints_dict: Dictionary containing the mapping between
            basenames and 2d headpoints numpy ndarrays
            (returned by get_headpoints_dict()).
        imgs_dir: Directory containing images (only their width and hight
            values are needed).
        args_dict: Dictionary containing required configuration values.
            The keys required for this function are 'part', 'sqr_side', 'knn',
            'max_knn_avg_dist', 'sigma_coef'.

    Returns:
        basename2dmap_dict: Dictionary containing the mapping between the 
        basenames and density maps (each density map has the same height 
        and width as the corresponding image).
    """
    side_len = args_dict['sqr_side']
    r = 1 + side_len // 2

    basename2dmap_dict = OrderedDict()

    for bn, points in tqdm(basename2headpoints_dict.items()):
        img_fpath = pjn(imgs_dir, bn[3:] + '.jpg')
        # bn[3:] means skipping the initial 'GT_' from the basename
        w, h = Image.open(img_fpath).size

        ## points.shape == (num_heads, 2)
        # `points` contains pairs (coord_along_w, coord_along_h) as floats

        neigh = NearestNeighbors(
            n_neighbors=(1 + args_dict['knn']),
            # each point^ is the closest one to itself
            metric='euclidean',
            n_jobs=-1
        )
        neigh.fit(points)
        knn_dists, knn_inds = neigh.kneighbors(points)

        dmap = np.zeros((h, w))

        for j, w_h_pair in enumerate(points):
            knn_dist_avg = knn_dists[j, 1:].mean()
            # excluding the point itself^ (zero distance)
            max_d = args_dict['max_knn_avg_dist']

            if (knn_dist_avg > max_d) and (args_dict['part'] == 'B'):
                knn_dist_avg = max_d

            sigma = args_dict['sigma_coef'] * knn_dist_avg
            one_head_gaussian = get_one_head_gaussian(side_len, r, sigma)
            one_head_sum = np.sum(one_head_gaussian)

            w_center = int(w_h_pair[0])
            h_center = int(w_h_pair[1])
            ##
            left = max(0, w_center - r)
            right = min(w, w_center + r)
            up = max(0, h_center - r)
            down = min(h, h_center + r)
            # ^ clip to the image boundaries
            ##
            left_g = left - w_center + r
            right_g = right - w_center + r
            up_g = up - h_center + r
            down_g = down - h_center + r
            # ^ one_head_gaussian must also be clipped to the image boundaries
            # after placing the gaussian center to the required location
            ##
            one_head_gaus_subset = one_head_gaussian[up_g:down_g,
                                                     left_g:right_g]
            dmap[up:down, left:right] += \
                one_head_gaus_subset / np.sum(one_head_gaussian)
            # seems that xhp uses division by np.sum(one_head_gaussian) ^ here
            # instead of np.sum(one_head_gaus_subset)!

        basename2dmap_dict[bn] = dmap
        #print(np.sum(dmap), points.shape[0])
        integral_eq_annot_num = (int(round(np.sum(dmap))) == points.shape[0])
        #assert integral_eq_annot_num
        # ^ Integral (sum) over the density map must be equal
        # to the annotated number of people
        # if dmap is normalized by np.sum(one_head_gaus_subset).
        # It will not hold if dmap is normalized by np.sum(one_head_gaussian).

    return basename2dmap_dict


def xhp_density_maps(xhp_dir):
    """
    Extract the density maps provided in the official reposity
    https://github.com/xhp-hust-2018-2011/S-DCNet

    Args:
        xhp_dir: Directory containing '*.mat' files.

    Returns:
        bname2dmap_dict: Dictionary containing the mapping between the 
        basenames and density maps from the official repository.
    """
    if not xhp_dir:
        print("--xhp-dir was not specified at the command line")
        return

    if not os.path.isdir(xhp_dir):
        print("directory '%s' not found" % xhp_dir)
        return

    xhp_mat_files = sorted(glob.glob(pjn(xhp_dir, "*.mat")))
    assert xhp_mat_files, "directory '%s' contains no '*.mat' files" % xhp_dir

    xhp_mat_bnames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in xhp_mat_files]
    bname2dmap_dict = OrderedDict()

    for fpath, bn in zip(xhp_mat_files, xhp_mat_bnames):
        mat = scipy.io.loadmat(fpath)
        bname2dmap_dict[bn] = mat['density_map']

    return bname2dmap_dict


def compare_to_xhp_dmaps(my_dmaps_dict, xhp_dmaps_dict, args_dict):
    """
    Compare the density maps obtained by generate_density_maps() with the
    density maps from the official repository.
    Save both sets of density maps as greyscale png images.

    Args:
        my_dmaps_dict: basenames <-> density maps mapping obtained by 
            the calling generate_density_maps().
        xhp_dmaps_dict: basenames <-> density maps mapping from the official
            repository.
        args_dict: Dictionary containing required configuration values.
            The key required for this function is 'part'.

    Returns:
        None.
    """
    if not xhp_dmaps_dict:
        return

    a = all([k1 == 'GT_' + k2
             for k1, k2 in zip(my_dmaps_dict.keys(), xhp_dmaps_dict.keys())])
    assert a, "image and ground truth file basenames are not consistent"

    abs_path = tempfile.mkdtemp(
        suffix=None,
        prefix='cmp_dmaps_part_%s_test_' % args_dict['part'],
        dir=os.getcwd())

    for dm1, (k2, dm2) in zip(my_dmaps_dict.values(), xhp_dmaps_dict.items()):
        skimage.io.imsave(
            pjn(abs_path, "%s_my.png" % k2),
            (dm1 / np.max(dm1) * 255).astype(np.uint8))
        skimage.io.imsave(
            pjn(abs_path, "%s_xhp.png" % k2),
            (dm2 / np.max(dm2) * 255).astype(np.uint8))


if __name__ == "__main__":
    args = parse_arguments()
    args_dict = vars(args)

    for t in ['train_data', 'test_data']:
        the_dir = pjn(
            args_dict['dataset_rootdir'],
            "part_" + args_dict['part'],
            t
        )
        imgs_dir = pjn(the_dir, "images")
        annot_dir = pjn(the_dir, "ground-truth")
        check_consist_imgs_annots(imgs_dir, annot_dir)

        bn2points_dict = get_headpoints_dict(annot_dir)

        print("generate_density_maps() call for part_%s %s"
              % (args_dict['part'], t[:-5]),
              flush=True)

        dmaps_dict = generate_density_maps(
            bn2points_dict,
            imgs_dir,
            args_dict)

        npz_name = "density_maps_part_%s_%s.npz" % (args_dict['part'], t[:-5])
        np.savez(npz_name, **dmaps_dict)

        if t == 'test_data':
            xhp_dmaps_dict = xhp_density_maps(args_dict['xhp_dir'])
            compare_to_xhp_dmaps(dmaps_dict, xhp_dmaps_dict, args_dict)
