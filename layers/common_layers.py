# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.init as init

class SparseAttModule(nn.Module):
    def __init__(self, N):
        super(SparseAttModule, self).__init__()
        self.forw_att = SparseAttentionBlock(N)
        self.back_att = SparseAttentionBlock(N)

    def forward(self, x, reverse=False):
        if not reverse:
            return self.forw_att(x)
        else:
            return self.back_att(x)

class ResidualUnit(nn.Module):
    """Sparse residual unit."""

    def __init__(self, N):
        super().__init__()
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N//2, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(in_channels=N//2, out_channels=N//2, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(in_channels=N//2, out_channels=N, kernel_size=1, stride=1, dimension=3),
        )
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        out = self.relu(out)
        return out

class SparseAttentionBlock(nn.Module):
    """Sparse Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto. 
    Adapted to sparse convolution domain for Point Clouds

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        self.conv_a = nn.Sequential(ResidualUnit(N), ResidualUnit(N), ResidualUnit(N))

        self.conv_b = nn.Sequential(
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=1, stride=1, dimension=3),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * ME.MinkowskiSigmoid()(b)
        out += identity
        return out

class SparseEnhModule(nn.Module):
    def __init__(self, nf):
        super(SparseEnhModule, self).__init__()
        self.forw_enh = SparseEnhBlock(nf)
        self.back_enh = SparseEnhBlock(nf)

    def forward(self, x, reverse=False):
        if not reverse:
            return self.forw_enh(x)
        else:
            return self.back_enh(x)

class SparseEnhBlock(nn.Module):
    def __init__(self, intermediate_channels):
        super(SparseEnhBlock, self).__init__()
        self.layers = nn.Sequential(
            SparseDenseBlock(3, intermediate_channels),
            ME.MinkowskiConvolution(in_channels=intermediate_channels,out_channels=intermediate_channels, kernel_size=1,stride=1,bias=True,dimension=3),
            ME.MinkowskiConvolution(in_channels=intermediate_channels,out_channels=intermediate_channels, kernel_size=3,stride=1,bias=True,dimension=3),
            ME.MinkowskiConvolution(in_channels=intermediate_channels,out_channels=intermediate_channels, kernel_size=1,stride=1,bias=True,dimension=3),
            SparseDenseBlock(intermediate_channels, 3)
        )

    def forward(self, x):
        y = self.layers(x)
        y._F = y.F * 0.2
        return x + y
    

class SparseDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, init='xavier', inner_channels=32, bias=True):
        super(SparseDenseBlock, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(in_channels=in_channels,out_channels=inner_channels,kernel_size=3, stride=1,bias=bias,dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=in_channels+inner_channels,out_channels=inner_channels,kernel_size=3, stride=1,bias=bias,dimension=3)
        self.conv3 = ME.MinkowskiConvolution(in_channels=in_channels+2*inner_channels,out_channels=inner_channels,kernel_size=3, stride=1,bias=bias,dimension=3)
        self.conv4 = ME.MinkowskiConvolution(in_channels=in_channels+3*inner_channels,out_channels=inner_channels,kernel_size=3, stride=1,bias=bias,dimension=3)
        self.conv5 = ME.MinkowskiConvolution(in_channels=in_channels+4*inner_channels,out_channels=out_channels,kernel_size=3, stride=1,bias=bias,dimension=3)
        self.lrelu = ME.MinkowskiLeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_me_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights_me([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights_me(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(ME.cat(x, x1)))
        x3 = self.lrelu(self.conv3(ME.cat((x, x1, x2))))
        x4 = self.lrelu(self.conv4(ME.cat((x, x1, x2, x3))))
        x5 = self.conv5(ME.cat((x, x1, x2, x3, x4)))

        return x5

def initialize_weights_me(layer_list,scale=1):
    if not isinstance(layer_list, list):
        layer_list = [layer_list]
    for layer in layer_list:
        if isinstance(layer,ME.MinkowskiConvolution):
            init.kaiming_normal_(layer.kernel,a=0,mode='fan_in')
            layer.kernel.data *= scale
            if layer.bias is not None:
                    layer.bias.data.zero_()

def initialize_weights_me_xavier(layer_list, scale=1):
    if not isinstance(layer_list, list):
        layer_list = [layer_list]
    for layer in layer_list:
        if isinstance(layer,ME.MinkowskiConvolution):
                init.xavier_normal_(layer.kernel)
                layer.kernel.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()