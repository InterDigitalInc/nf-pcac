# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import torch.nn as nn
import MinkowskiEngine as ME

def HyperAnalysisTransform(conv_filters,conv_kernel_size,conv_strides):

    layer_list=[]

    n_layers = len(conv_filters)-1

    for i in range(n_layers):
        layer_list.append(ME.MinkowskiConvolution(
                                in_channels = conv_filters[i],
                                out_channels = conv_filters[i+1], 
                                kernel_size = conv_kernel_size[i],
                                stride = conv_strides[i],
                                bias = True,
                                dimension=3,
                                ))
        if (i!=n_layers-1 and i%2==0):
            layer_list.append(ME.MinkowskiReLU(inplace=True))

    return nn.Sequential(*layer_list)

def HyperSynthesisTransform(conv_filters,conv_kernel_size,conv_strides):

    layer_list = []
    n_layers = len(conv_filters)-1

    for i in range(n_layers):
        if (i%2==0) and (i!=n_layers-1):
            layer_list.append(ME.MinkowskiConvolutionTranspose(in_channels = conv_filters[i],
                                                out_channels = conv_filters[i+1],
                                                kernel_size = conv_kernel_size[i],
                                                stride = conv_strides[i],
                                                bias = True,
                                                dimension=3,
                                                ))
        else:
            layer_list.append(ME.MinkowskiConvolution(in_channels = conv_filters[i],
                                                out_channels = conv_filters[i+1],
                                                kernel_size = conv_kernel_size[i],
                                                stride = conv_strides[i],
                                                bias = True,
                                                dimension=3,
                                                ))
        
        if (i!=n_layers-1 and i%2==1):
            layer_list.append(ME.MinkowskiReLU(inplace=True))

    return nn.Sequential(*layer_list)
