# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import numpy as np

def _blend(pc1, pc2, ratio: float):
    """
    Implemented based on Pytorch's implementation.
    Available: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L146
    Adapted for our PC case.
    """
    ratio = float(ratio)
    bound = 1
    return np.clip((ratio * pc1 + (1.0 - ratio) * pc2), 0, bound)

def rgb_to_grayscale(pc):
    l_pc = 0.2989 * pc[:,0] + 0.587 * pc[:,1] + 0.114 * pc[:,2]
    return np.expand_dims(l_pc,1)

def _rgb2hsv(pc):
    """
    Implemented based on Pytorch's implementation.
    Available: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L146
    Adapted for our PC case.
    """
    r, g, b = pc[:,0], pc[:,1], pc[:,2]

    maxc = np.max(pc, axis=1)
    minc = np.min(pc, axis=1)

    eqc = maxc == minc

    cr = maxc - minc

    ones = np.ones_like(maxc)
    s = cr / np.where(eqc, ones, maxc)

    cr_divisor = np.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = np.fmod((h / 6.0 + 1.0), 1.0)
    return np.stack((h, s, maxc),axis=1)

def _hsv2rgb(pc):
    """
    Implemented based on Pytorch's implementation.
    Available: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L146
    Adapted for our PC case.
    """
    h, s, v = pc[:,0], pc[:,1], pc[:,2]
    i = np.floor(h * 6.0)
    f = (h * 6.0) - i

    p = np.clip((v * (1.0 - s)), 0.0, 1.0)
    q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
    t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = np.expand_dims(i,0) == np.expand_dims(np.arange(6),axis=(1,2))

    a1 = np.stack((v, q, p, p, t, v), axis=1)
    a2 = np.stack((t, v, v, q, p, p), axis=1)
    a3 = np.stack((p, p, t, v, v, q), axis=1)
    a4 = np.stack((a1, a2, a3), axis=2)

    return np.einsum("...ijk, ...kix -> ...jkx", mask, a4)

def invert(pc):
    return 1 - pc

def adjust_brightness(pc, brightness_factor: float):
    """
    Implemented based on Pytorch's implementation.
    Available: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L146
    Adapted for our PC case.
    """
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")

    return _blend(pc, np.zeros_like(pc), brightness_factor)

def adjust_contrast(pc, contrast_factor: float):
    """
    Implemented based on Pytorch's implementation.
    Available: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L146
    Adapted for our PC case.
    """
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")

    mean = np.mean(pc)
    
    return _blend(pc, mean, contrast_factor)

def adjust_saturation(pc, saturation_factor: float):
    """
    Implemented based on Pytorch's implementation.
    Available: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L146
    Adapted for our PC case.
    """
    if saturation_factor < 0:
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")

    return _blend(pc, rgb_to_grayscale(pc), saturation_factor)

def adjust_hue(pc, hue_factor: float):
    """
    Implemented based on Pytorch's implementation.
    Available: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L146
    Adapted for our PC case.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    pc = _rgb2hsv(pc)
    h, s, v = pc[:,0], pc[:,1], pc[:,2]
    h = (h + hue_factor) % 1.0
    pc = np.stack((h, s, v),axis=1)
    pc_hue_adj = _hsv2rgb(pc)

    return np.squeeze(pc_hue_adj,axis=0)

def solarize(pc, threshold: float):

    inverted_pc = invert(pc)
    return np.where(pc >= threshold, inverted_pc, pc)

def channel_swap(Colors, permutation):
    return Colors[:, permutation]

def color_shift(Colors, channel, value):
    return Colors + (value*channel)