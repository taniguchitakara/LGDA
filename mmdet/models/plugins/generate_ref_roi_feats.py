
from torchvision.ops import roi_align
import torch.nn as nn
import torch
import torch.nn.functional as F
import mmcv
import numpy as np
import cv2

import torch

def add_gaussian_noise(input_data, mean=0, std=None):
    """
    Add Gaussian noise to a PyTorch tensor or tuple of tensors.

    Args:
        input_data (torch.Tensor or tuple of torch.Tensor): Input data.
        mean (float): Mean of the Gaussian noise (default is 0).
        std (torch.Tensor or float): Standard deviation of the Gaussian noise (default is 0).

    Returns:
        torch.Tensor or tuple of torch.Tensor: Data with added Gaussian noise.
    """
    if isinstance(input_data, torch.Tensor):
        noise = torch.randn_like(input_data)
        if isinstance(std, torch.Tensor):
            noise = noise * std + mean
        else:
            noise = noise * std + mean
        return input_data + noise
    elif isinstance(input_data, tuple):
        noisy_data = tuple(tensor + torch.randn_like(tensor) * std + mean if isinstance(std, float) 
                           else tensor + torch.randn_like(tensor) * std[i] + mean 
                           for i, tensor in enumerate(input_data))
        return noisy_data
    else:
        raise ValueError("Input must be a torch.Tensor or a tuple of torch.Tensors.")


#old add gaussian noise function

def add_gaussian_noise_old(input_data, mean=0, std=0):
    """
    Add Gaussian noise to a PyTorch tensor or tuple of tensors.

    Args:
        input_data (torch.Tensor or tuple of torch.Tensor): Input data.
        mean (float): Mean of the Gaussian noise (default is 0).
        std (float): Standard deviation of the Gaussian noise (default is 10).

    Returns:
        torch.Tensor or tuple of torch.Tensor: Data with added Gaussian noise.
    """
    if isinstance(input_data, torch.Tensor):
        noise = torch.randn_like(input_data) * std + mean
        return input_data + noise
    elif isinstance(input_data, tuple):
        noisy_data = tuple(tensor + torch.randn_like(tensor) * std + mean for tensor in input_data)

        #for tensor in noisy_data:
        #    tensor.requires_grad = True
        return noisy_data
    else:
        raise ValueError("Input must be a torch.Tensor or a tuple of torch.Tensors.")

def forward_fuse(feats):
    feats = list(feats)
    feats[0] = feats[0].unsqueeze(1)
    for i in range(1, len(feats)):
        feats[i] = F.interpolate(feats[i], scale_factor=2 ** i, mode='nearest')
        feats[i] = feats[i].unsqueeze(1)
    feats = torch.cat(feats, dim=1)
    feats = feats.mean(dim=1)
    return feats


def generate_ref_roi_feats(rf_feat, bbox):
    ref_fuse_feats = forward_fuse(rf_feat)
    roi_feats = []
    for j in range(bbox.shape[0]):
        roi_feat = roi_align(ref_fuse_feats[j].unsqueeze(0), [bbox[j] / 4], [7, 7])
        roi_feats.append(roi_feat)
    roi_feats = torch.cat(roi_feats, dim=0)
    return roi_feats



