# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

from typing import Any
import numpy as np
import torchvision
import torch
import math as m
import random

from data_augmentation.functions import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation, channel_swap, color_shift, solarize

class ChannelSwap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.permutations = [[0,1,2],
                             [0,2,1],
                             [1,0,2],
                             [1,2,0],
                             [2,0,1],
                             [2,1,0]]

    def get_params(self):
        return np.random.randint(6)

    def forward(self, V, Colors):
        
        perm = self.get_params()
        Colors = channel_swap(Colors,self.permutations[perm])

        return V, Colors

class ColorShift(torch.nn.Module):
    """
    Inspired by WU, Ren, YAN, Shengen, SHAN, Yi, et al. Deep image: Scaling up image recognition. arXiv preprint arXiv:1501.02876, 2015, vol. 7, no 8, p. 4.

    "Specifically, for each image, we generate three Boolean values to determine if the R, G and B channels should be altered, respectively. If one channel should be altered, we add a random integer ranging from -20 to +20 to that channel."

    """
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def get_params(self,shift):
        channel = np.random.randint(2,size=[1,3])
        value = np.random.randint(-shift,shift,size=[1,3])/255
        return channel, value

    def __call__(self, V, Colors):
        
        channel, value = self.get_params(self.shift)

        Colors = color_shift(Colors,channel,value)

        return V, Colors
    
class SparseRandomSolarize(torchvision.transforms.RandomSolarize):

    def get_random_threshold(self):
        return float(torch.empty(1).uniform_(0, self.threshold))

    def forward(self, V, Colors):
        if np.random.randint(2) < self.p:
            return V, solarize(Colors,self.get_random_threshold())
        return V, Colors


# class RandomSharpness():

class SparseColorJitter(torchvision.transforms.ColorJitter):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """
    def forward(self, V, Colors):

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                Colors = adjust_brightness(Colors, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                Colors = adjust_contrast(Colors, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                Colors = adjust_saturation(Colors, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                Colors = adjust_hue(Colors, hue_factor)

        return V, Colors


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])


class RotationAugmentation(torch.nn.Module):
    def __init__(self, min_rotation=0, max_rotation=m.pi, step=m.pi/18) -> None:
        super().__init__()
        self.rotation_range = np.arange(start=min_rotation, stop=max_rotation ,step=step)
        self.R = [Rx,Ry,Rz]

    def get_params(self):
        axis_to_rotate = [bool(random.getrandbits(1)),bool(random.getrandbits(1)),bool(random.getrandbits(1))]
        rotation_angle = [np.random.choice(self.rotation_range),np.random.choice(self.rotation_range),np.random.choice(self.rotation_range)]
        return axis_to_rotate, rotation_angle

    def forward(self, V, Colors):
        ori_points = V
        # Center the point cloud (starting at origin)
        points=ori_points-np.min(ori_points,axis=0)
        # Define the axis to rotate and how to rotate it
        axis_to_rotate, rotation_angle = self.get_params()
        for i in range(3):
            if axis_to_rotate[i]:
                R_coef=self.R[i](rotation_angle[i])
                points = points*R_coef

                points=points-np.min(points,axis=0)
                points=np.round(points)

        return points, Colors