#!/usr/bin/python3

import sys
from sys import exit as e
import os
import os.path
from os.path import join as pjn
import hydra
import q
import numpy as np
import torch

from labels_counts_utils import make_label2count_list
import ShanghaiTech_dataset as dtst
from SDCNet import SDCNet


@hydra.main(config_path="conf/config_train_val_test.yaml")
def main(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    print(f"  Exporting the checkpoint "
          f"'{cfg.test.trained_ckpt_for_inference}'")
    ckpt_path = pjn(orig_cwd, cfg.test.trained_ckpt_for_inference)
    loaded_struct = torch.load(ckpt_path)

    interval_bounds, label2count_list = make_label2count_list(cfg)
    model = SDCNet(
        label2count_list,
        cfg.model.supervised,
        load_pretr_weights_vgg=False)
    model.load_state_dict(loaded_struct['model_state_dict'], strict=True)
    
    batch_size = 1
    x = torch.randn(batch_size, 3, 64*1, 64*1, requires_grad=False)
    
    if not cfg.resources.disable_cuda and torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()

    p1, ext = os.path.splitext(ckpt_path)
    dir_, bname = os.path.split(p1)
    
    try:
        torch.onnx.export(model, x, p1 + ".onnx", opset_version=13)
    except:
        print("  (!) Failed to export the checkpoint to ONNX format")
    else:
        print("  Successfully exported the checkpoint to ONNX format")

    try:
        traced_script_module = torch.jit.trace(model, x)
        traced_script_module.save(bname + "_jit_trace.pt")
    except:
        print("  (!) Failed to export the checkpoint to 'torch jit trace' format")
    else:
        print("  Successfully exported the checkpoint to 'torch jit trace' format")

    try:
        script_module = torch.jit.script(model)
        script_module.save(bname + "_jit_script.pt")
    except:
        print("  (!) Failed to export the checkpoint to 'torch jit script' format")
    else:
        print("  Successfully exported the checkpoint to 'torch jit script' format")


if __name__ == "__main__":
    main()
