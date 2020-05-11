import sys
import os
import os.path
from sys import exit as e

from bisect import bisect_left
import numpy as np
import torch
import q


def apply_label2count(cls_labels_tensor, cls_label2count_tensor):
    """
    Function for obtaining a tensor containing count values
    from a tensor containing class labels.

    Implementation details: torch.index_select() is applied to the flattened
    versions of the tensors.

    Args:
        cls_labels_tensor: Tensor (of arbitrary shape in general)
            containing class labels (integers).
        cls_label2count_tensor: Tensor containing 1-to-1 mapping
            between a scalar label (integer) to a scalar count value (float).

    Returns:
        Tensor containing count values (instead of labels).
        It has the same shape as `cls_labels_tensor`.
    """
    orig_shape = cls_labels_tensor.shape
    t = torch.index_select(
        cls_label2count_tensor,  # input
        dim=0,
        index=cls_labels_tensor.reshape((-1,))
    )
    # ^ DO NOT specify the 1st argument as input=<smth>!
    # TorchScript will throw `RuntimeError: Arguments for call are not valid`.
    # aten::index_select(Tensor self, int dim, Tensor index) -> (Tensor):
    # Argument self not provided.

    return t.reshape(orig_shape)


def apply_count2label(counts_tensor, interval_bounds):
    """
    Function for obtaining a tensor containing class labels
    from a tensor containing count values (inverse to apply_label2count()).

    Implementation details: bisect.bisect_left() is called on the sorted 
    interval bounds (for count values) and the passed count values.

    Args:
        counts_tensor: Tensor containing count values (floats).
        interval_bounds: Interval boundaries for the count values (floats).

    Returns:
        Tensor containing class labels (instead count values).
        It has the same shape as `counts_tensor`.
    """
    orig_shape = counts_tensor.shape
    labels_list = []

    for c in counts_tensor.reshape((-1,)).tolist():
        labels_list.append(bisect_left(interval_bounds, c))

    result = np.array(labels_list).reshape(orig_shape)
    return torch.from_numpy(result)


def make_label2count_list(cfg):
    """
    Construct the mapping between the class labels (int) and count values
    (float).
    Interval boundaries are the base for both class labels and count values.
    Class labels are simply consecutive indices (zero-based) of the adjacent 
    intervals. Count values are middle points of the intervals (except for the
    rightmost interval which is semi-open [C, +inf); the left boundary C
    is chosen as the count value in this case).

    Args:
        args_dict: Dictionary containing required configuration values.
            The keys required for this function are 'num_intervals',
            'interval_step', 'partition_method'.

    Returns:
        Interval boundaries; list with the count values (their indices are
        the class labels).
    """
    s = cfg.model.interval_step
    Cmax = cfg.dataset.num_intervals

    numpoints = int((0.45 - 0.05) / 0.05) + 1
    add_for_two_linear = np.array([])
    if cfg.model.partition_method == 2:
        add_for_two_linear = np.linspace(0.05, 0.45, numpoints)

    numpoints = int((Cmax - s) / s) + 1
    bounds = np.linspace(s, Cmax, numpoints)

    very_1st_bnd = np.array([1e-6, ])
    interval_bounds = np.concatenate(
        [very_1st_bnd, add_for_two_linear, bounds])

    # tranform interval endpoints to count values
    bnds = interval_bounds.tolist()
    label2count_list = [0.0, ]
    # ^ label is the index, count is the value of the list element

    for i in range(len(bnds) - 1):
        label2count_list.append((bnds[i] + bnds[i+1]) / 2.0)
    label2count_list.append(bnds[-1])
    #print("num_classes =", len(label2count_list))

    l = interval_bounds
    ascending = [l[i] <= l[i+1] for i in range(len(l)-1)]
    assert all(ascending)

    l = label2count_list
    ascending = [l[i] <= l[i+1] for i in range(len(l)-1)]
    assert all(ascending)

    return interval_bounds, label2count_list
