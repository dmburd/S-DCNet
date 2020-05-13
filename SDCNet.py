import sys
import os
import os.path
from sys import exit as e
import copy
import re
import math
import numpy as np
from collections import OrderedDict
import q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary

from labels_counts_utils import apply_label2count, make_label2count_list


def make_layers_vgg(cfg, in_ch=3, use_batch_norm=False):
    """
    Code borrowed from torchvision/models/vgg.py
    """
    layers = []

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_ch, v, kernel_size=3, padding=1)
            if use_batch_norm:
                layers.extend(
                    [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_ch = v

    return nn.Sequential(*layers)


def fully_conv_classif(in_ch, num_classes):
    """
    Used for counter classification part of the network,
    for division decider and for upsampler.
    """
    layers = [
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_ch, in_ch, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, num_classes, kernel_size=1)
    ]
    return nn.Sequential(*layers)


def one_conv(in_ch, out_ch, use_batch_norm=False):
    """
    Used for upsampling (class Up()).
    """
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def double_conv(in_ch, out_ch, use_batch_norm=False):
    """
    Used for upsampling (class Up()).
    """
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class Up(nn.Module):
    """
    UNet-style upsampling. 
    See https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)
            self.conv1 = one_conv(up_in_ch, up_out_ch)
        else:
            self.up = nn.ConvTranspose2d(
                up_in_ch, up_out_ch, kernel_size=2, stride=2)

        self.conv2 = double_conv(cat_in_ch, cat_out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1(self.up(x1))
        else:
            x1 = self.up(x1)

        ## input is NCHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # ^ RuntimeError: Failed to export an ONNX attribute,
        # since it's not constant, please try to make things
        # (e.g., kernel size) static if possible

        # Fortunately, padding is not required in this case,
        # x1 and x2 have the same height and width for S-DCNet

        x = torch.cat([x2, x1], dim=1)
        return self.conv2(x)


def spatial_2x2_softmax(x):
    out1 = F.unfold(x, kernel_size=2, stride=2)
    out2 = F.softmax(out1, dim=1)
    out3 = F.fold(out2, (x.shape[2], x.shape[3]), kernel_size=2, stride=2)
    return out3
    #UserWarning: ONNX export failed on ATen operator im2col 
    # because torch.onnx.symbolic_opset11.im2col does not exist
    # (when trying to run torch.onnx.export() )


class SDCNet(nn.Module):
    """
    The whole architecture of S-DCNet / SS-DCNet.
    """

    def __init__(
            self,
            label2count_list,
            supervised=False, # False for S-DCNet, True for SS-DCNet
            load_pretr_weights_vgg=False):
        super(SDCNet, self).__init__()

        # vgg16, corresponds to cfg['D'] from torchvision/models/vgg.py
        self.conv1_features = make_layers_vgg([64, 64, 'M'], in_ch=3)
        self.conv2_features = make_layers_vgg([128, 128, 'M'], in_ch=64)
        self.conv3_features = make_layers_vgg([256, 256, 256, 'M'], in_ch=128)
        self.conv4_features = make_layers_vgg([512, 512, 512, 'M'], in_ch=256)
        self.conv5_features = make_layers_vgg([512, 512, 512, 'M'], in_ch=512)

        self.label2count_tensor = torch.tensor(
            label2count_list, dtype=torch.float32)

        num_classes = len(label2count_list)
        self.count_interval_classif = fully_conv_classif(512, num_classes)

        ## upsampling (UNet-like)
        self.up_from_5_to_4 = Up(
            up_in_ch=512, up_out_ch=256, cat_in_ch=(256+512), cat_out_ch=512)

        self.division_decider = fully_conv_classif(512, 1)
        self.supervised = supervised
        self.upsampler = fully_conv_classif(512, 1)
        
        self.up_from_4_to_3 = Up(
            up_in_ch=512, up_out_ch=256, cat_in_ch=(256+256), cat_out_ch=512)

        self._initialize_weights()

        if load_pretr_weights_vgg:
            pretr_dict = torchvision.models.vgg16(pretrained=True).state_dict()
            this_net_dict = self.state_dict()
            this_net_keys = list(this_net_dict.keys())

            for i, (pretr_key, pretr_tensor_val) in enumerate(pretr_dict.items()):
                # pretrained vgg16 keys start with 'features' or with 'classifier'
                if 'features' in pretr_key:
                    this_net_tensor_val = this_net_dict[this_net_keys[i]]
                    assert this_net_tensor_val.shape == pretr_tensor_val.shape
                    this_net_tensor_val.data = pretr_tensor_val.data.clone()
                    #print(pretr_key, pretr_tensor_val.shape)
                else:
                    break

            self.load_state_dict(this_net_dict)

    def forward(self, x):
        x = self.conv1_features(x)
        x = self.conv2_features(x)
        x = self.conv3_features(x)
        conv3_feat = x
        # ^ will be used for obtaining F2
        x = self.conv4_features(x)
        conv4_feat = x
        # ^ will be used for obtaining F1
        x = self.conv5_features(x)
        conv5_feat = x

        F0 = conv5_feat
        cls0_logits = self.count_interval_classif(F0)
        cls0 = torch.argmax(cls0_logits, dim=1, keepdim=True)
        C0 = apply_label2count(cls0, self.label2count_tensor.to(cls0.device))

        F1 = self.up_from_5_to_4(F0, conv4_feat)
        W1 = torch.sigmoid(self.division_decider(F1))
        if self.supervised:
            U1 = spatial_2x2_softmax(self.upsampler(F1))
        else:
            U1 = torch.ones_like(W1) / 4.0
        
        cls1_logits = self.count_interval_classif(F1)
        cls1 = torch.argmax(cls1_logits, dim=1, keepdim=True)
        C1 = apply_label2count(cls1, self.label2count_tensor.to(cls1.device))

        F2 = self.up_from_4_to_3(F1, conv3_feat)
        W2 = torch.sigmoid(self.division_decider(F2))
        if self.supervised:
            U2 = spatial_2x2_softmax(self.upsampler(F2))
        else:
            U2 = torch.ones_like(W2) / 4.0

        cls2_logits = self.count_interval_classif(F2)
        cls2 = torch.argmax(cls2_logits, dim=1, keepdim=True)
        C2 = apply_label2count(cls2, self.label2count_tensor.to(cls2.device))

        krn = torch.ones((1, 1, 2, 2)).to(C0.device)
        # ^ kernel for conv_transpose2d
        #   (used for calculate Kronecker product C0 [kron_prod] 1(2x2)
        C0_x2x2 = F.conv_transpose2d(C0, krn, stride=2)
        C0_redistr_2x2 = C0_x2x2 * U1
        DIV1 = (1.0 - W1) * C0_redistr_2x2 + W1 * C1
        DIV1_x2x2 = F.conv_transpose2d(DIV1, krn, stride=2)
        DIV1_redistr_2x2 = DIV1_x2x2 * U2
        DIV2 = (1.0 - W2) * DIV1_redistr_2x2 + W2 * C2

        tuple_for_loss_calc = (
            cls0_logits, cls1_logits, cls2_logits, DIV2,
            U1, U2, W1, W2)

        return tuple_for_loss_calc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    """
    Code for debugging.
    Create a model instance and save it in several supported formats.
    """
    class Object(object):
        pass
    
    class Cfg:
        # imitates config from hydra (only 3 parameters from the full config
        # are required here to instantiate SDCNet)
        def __init__(self, num_intervals, interval_step, partition_method):
            self.dataset = Object()
            self.dataset.num_intervals = num_intervals
            self.model = Object()
            self.model.interval_step = interval_step
            self.model.partition_method = partition_method
            
    cfg = Cfg(22, 0.5, 2) # for part_A
    #cfg = Cfg(7, 0.5, 2) # for part_B
    interval_bounds, label2count_list = make_label2count_list(cfg)

    # ================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sdcnet_instance = SDCNet(
        label2count_list,
        supervised=True,
        load_pretr_weights_vgg=True
    ).to(device)

    batch_size = 1
    h = 64 * 3
    w = 64 * 2
    summary(sdcnet_instance, (3, h, w))
    
    x = torch.randn(batch_size, 3, h, w, requires_grad=False).to(device)
    out_list = sdcnet_instance(x)
    shapes_list = [str(one_featmap.shape) for one_featmap in out_list]
    print("\n".join(shapes_list))
    
    # save in several possible ways
    torch.save(sdcnet_instance.state_dict(), "sdcnet_state_dict.pth")

    #torch.save(sdcnet_instance, "sdcnet_full_model.pth")
    # ^ UserWarning: Couldn't retrieve source code for container of type Conv2d
    #   (torch.__version__ == '1.3.0')

    try:
        torch.onnx.export(sdcnet_instance, x, "sdcnet.onnx", opset_version=11)
    except:
        print("  torch.onnx.export() call failed")

    traced_script_module = torch.jit.trace(sdcnet_instance, x)
    traced_script_module.save("traced_sdcnet_model.pt")

    script_module = torch.jit.script(sdcnet_instance)
    script_module.save("sdcnet_script_model.pt")
