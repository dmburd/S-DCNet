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
import hydra
import tempfile
import time
from tqdm import tqdm
from multiprocessing import Process, Manager

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


def check_consist_imgs_annots(imgs_dir, annot_dir):
    """
    Check the correspondence between the images and annotations.
    `imgs_dir` must contain '*.jpg' files, `annot_dir` must contain 
    the same number of '*.mat' files, the basenames of the files must
    differ only by the leading 'GT_' substring
    (in the '*.mat' file basenames).
    """
    if not os.path.isdir(imgs_dir):
        raise FileNotFoundError(f"images directory '{imgs_dir}' is not found")
    
    jpg_files = sorted(glob.glob(pjn(imgs_dir, "*.jpg")))
    if not jpg_files:
        raise FileNotFoundError(
            f"directory '{imgs_dir}' contains no '*.jpg' files")

    jpg_basenames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in jpg_files]

    if not os.path.isdir(annot_dir):
        raise FileNotFoundError(
            f"annotations directory '{annot_dir}' is not found")
    
    mat_files = sorted(glob.glob(pjn(annot_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"directory '{annot_dir}' contains no '*.mat' files")

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


def generate_density_maps(
        basename2headpoints_dict_part,
        basename2dmap_dict,
        imgs_dir,
        cfg):
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
    `cfg.one_headpoint_dmap.knn` (3 by default) nearest neighbors for that 
    point are found and average distance to them is calculated. That average 
    distance is capped by the constant pre-defined value 
    `cfg.one_headpoint_dmap.max_knn_avg_dist` (50.0 by default) for ShanghaiTech 
    part_B dataset (the average distance is not capped for ShanghaiTech
    part_A). The `sigma` (Gaussian RMS width) is the product of the average
    distance and a pre-defined constant `cfg.one_headpoint_dmap.sigma_coef`
    (0.3 by default).
    The sum of the density map values across the whole image area must be
    equal to the number of annotated heads.

    Args:
        basename2headpoints_dict_part: Part of the dictionary containing 
            the mapping between basenames and 2d headpoints numpy ndarrays
            (returned by get_headpoints_dict()).
        basename2dmap_dict: Dictionary that will be filled with the mapping
            between the basenames and density maps (each density map has
            the same height and width as the corresponding image).
        imgs_dir: Directory containing images (only their width and hight
            values are needed).
        cfg: the global configuration (hydra).

    Returns:
    """
    side_len = cfg.one_headpoint_dmap.sqr_side
    r = 1 + side_len // 2

    for bn, points in basename2headpoints_dict_part.items():
        img_fpath = pjn(imgs_dir, bn[3:] + '.jpg')
        # bn[3:] means skipping the initial 'GT_' from the basename
        w, h = Image.open(img_fpath).size

        ## points.shape == (num_heads, 2)
        # `points` contains pairs (coord_along_w, coord_along_h) as floats

        neigh = NearestNeighbors(
            n_neighbors=(1 + cfg.one_headpoint_dmap.knn),
            # each point ^ is the closest one to itself
            metric='euclidean',
            n_jobs=-1)
        neigh.fit(points)
        knn_dists, knn_inds = neigh.kneighbors(points)

        dmap = np.zeros((h, w))

        for j, w_h_pair in enumerate(points):
            knn_dist_avg = knn_dists[j, 1:].mean()
            # excluding the point itself^ (zero distance)
            max_d = cfg.one_headpoint_dmap.max_knn_avg_dist

            if (knn_dist_avg > max_d) and (cfg.dataset.part == 'B'):
                knn_dist_avg = max_d

            sigma = cfg.one_headpoint_dmap.sigma_coef * knn_dist_avg
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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    

def generate_density_maps_paral(basename2headpoints_dict, imgs_dir, cfg):
    basenames = list(basename2headpoints_dict.keys())
    random.shuffle(basenames)
    
    chunk_size = int(math.ceil(len(basenames) / cfg.resources.num_proc))
    basenames_chunks = chunks(basenames, chunk_size)
    
    manager = Manager()
    basename2dmap_dict = manager.dict()
    procs = []
    
    for basenames_chunk in basenames_chunks:
        bn2hp_dict_part = {
            bn: basename2headpoints_dict[bn] for bn in basenames_chunk}
        p = Process(
            target=generate_density_maps, 
            args=(bn2hp_dict_part, basename2dmap_dict, imgs_dir, cfg)
        )
        p.start()
        procs.append(p)
    
    for p in procs:
        p.join()
    
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
        print(f"  Directory '{xhp_dir}' (that is expected to contain "
              f"xhp density maps) not found")
        return

    xhp_mat_files = sorted(glob.glob(pjn(xhp_dir, "*.mat")))
    if not xhp_mat_files:
        print(f"  Directory '{xhp_dir}' (that is expected to contain "
              f"xhp density maps) contains no '*.mat' files")
        return

    xhp_mat_bnames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in xhp_mat_files]
    bname2dmap_dict = OrderedDict()

    for fpath, bn in zip(xhp_mat_files, xhp_mat_bnames):
        mat = scipy.io.loadmat(fpath)
        bname2dmap_dict[bn] = mat['density_map']

    return bname2dmap_dict


def compare_to_xhp_dmaps(my_dmaps_dict, xhp_dmaps_dict, cfg):
    """
    Compare the density maps obtained by generate_density_maps() with the
    density maps from the official repository.
    Save both sets of density maps as greyscale png images.

    Args:
        my_dmaps_dict: basenames <-> density maps mapping obtained by 
            the calling generate_density_maps().
        xhp_dmaps_dict: basenames <-> density maps mapping from the official
            repository.
        cfg: the global configuration (hydra).

    Returns:
        None.
    """
    if not xhp_dmaps_dict:
        return

    my_dmaps_dict_keys = sorted(list(my_dmaps_dict.keys()))
    xhp_dmaps_dict_keys = sorted(list(xhp_dmaps_dict.keys()))
    a = (len(my_dmaps_dict_keys) == len(xhp_dmaps_dict_keys))
    assert a, "image and ground truth file basenames are not consistent"
    a = all([k1 == 'GT_' + k2
             for k1, k2 in zip(my_dmaps_dict_keys, xhp_dmaps_dict_keys)])
    assert a, "image and ground truth file basenames are not consistent"

    abs_path = tempfile.mkdtemp(
        suffix=None,
        prefix=f"cmp_dmaps_part_{cfg.dataset.part}_test_",
        dir=os.getcwd())

    print(f"  The density maps to be visually compared are being saved to "
          f"{os.path.relpath(abs_path, os.getcwd())}")
    
    def func(
            my_dmaps_dict_keys,
            xhp_dmaps_dict_keys,
            my_dmaps_dict,
            xhp_dmaps_dict,
            abs_path):
        for my_k, xhp_k in zip(my_dmaps_dict_keys, xhp_dmaps_dict_keys):
            dm1 = my_dmaps_dict[my_k]
            dm2 = xhp_dmaps_dict[xhp_k]
            skimage.io.imsave(
                pjn(abs_path, f"{xhp_k}_my.png"),
                (dm1 / np.max(dm1) * 255).astype(np.uint8))
            skimage.io.imsave(
                pjn(abs_path, f"{xhp_k}_xhp.png"),
                (dm2 / np.max(dm2) * 255).astype(np.uint8))

    chunk_size = int(math.ceil(
        len(my_dmaps_dict_keys) / cfg.resources.num_proc))
    my_dmaps_dict_keys_chunks = chunks(my_dmaps_dict_keys, chunk_size)
    xhp_dmaps_dict_keys_chunks = chunks(xhp_dmaps_dict_keys, chunk_size)
    
    procs = []
    zipped_chunks = zip(my_dmaps_dict_keys_chunks, xhp_dmaps_dict_keys_chunks)
    
    if True: #cfg.resources.num_proc == 1:
        for ch1, ch2 in zipped_chunks:
            func(ch1, ch2, my_dmaps_dict, xhp_dmaps_dict, abs_path)
    else:
        for ch1, ch2 in zipped_chunks:
            p = Process(
                target=func, 
                args=(ch1, ch2, my_dmaps_dict, xhp_dmaps_dict, abs_path)
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


@hydra.main(config_path="conf/config_density_maps.yaml")
def main(cfg):
    for t in ['train_data', 'test_data']:
        the_dir = pjn(
            cfg.dataset.dataset_rootdir,
            f"part_{cfg.dataset.part}",
            t)
        imgs_dir = pjn(the_dir, "images")
        annot_dir = pjn(the_dir, "ground-truth")
        check_consist_imgs_annots(imgs_dir, annot_dir)

        bn2points_dict = get_headpoints_dict(annot_dir)

        print(f"  Calling generate_density_maps_paral() for "
              f"part_{cfg.dataset.part} {t[:-5]}... ",
              end='',
              flush=True)

        if cfg.resources.num_proc == 1:
            # single-process mode for debugging
            dmaps_dict = {}
            generate_density_maps(bn2points_dict, dmaps_dict, imgs_dir, cfg)
        else:
            dmaps_dict = generate_density_maps_paral(
                bn2points_dict,
                imgs_dir,
                cfg)

        print(f"Done")
        
        npz_name = f"density_maps_part_{cfg.dataset.part}_{t[:-5]}.npz"
        print(f"  Saving the file {npz_name}")
        np.savez(npz_name, **dmaps_dict)

        if t == 'test_data':
            xhp_dmaps_dict = xhp_density_maps(cfg.dataset.xhp_gt_dmaps_dir)
            compare_to_xhp_dmaps(dmaps_dict, xhp_dmaps_dict, cfg)


if __name__ == "__main__":
    main()
