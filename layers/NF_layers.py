# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import numpy as np
import torch
import torch.nn as nn
from layers.common_layers import *
import torch.nn.functional as F
from compressai.layers import *
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiFunctional import _wrap_tensor

class SparseCouplingLayer(nn.Module):
    def __init__(self, split_len1, split_len2, kernal_size, clamp=1.0):
        super(SparseCouplingLayer, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.G1 = SparseBottleneck(self.split_len1, self.split_len2, kernal_size)
        self.G2 = SparseBottleneck(self.split_len2, self.split_len1, kernal_size)
        self.H1 = SparseBottleneck(self.split_len1, self.split_len2, kernal_size)
        self.H2 = SparseBottleneck(self.split_len2, self.split_len1, kernal_size)

    def forward(self, x, reverse=False):
        # print(x.shape)
        x1, x2 = (x.F.narrow(1, 0, self.split_len1), x.F.narrow(1, self.split_len1, self.split_len2))
        x1 = _wrap_tensor(x,x1)
        x2 = _wrap_tensor(x,x2)

        if not reverse:
            y1 = x1.F.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(x2).F) * 2 - 1) )) + self.H2(x2).F
            y1 = _wrap_tensor(x,y1)

            y2 = x2.F.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(y1).F) * 2 - 1) )) + self.H1(y1).F
            y2 = _wrap_tensor(x,y2)

        else:
            y2 = (x2.F - self.H1(x1).F).div(torch.exp( self.clamp * (torch.sigmoid(self.G1(x1).F) * 2 - 1) ))
            y2 = _wrap_tensor(x,y2)

            y1 = (x1.F - self.H2(y2).F).div(torch.exp( self.clamp * (torch.sigmoid(self.G2(y2).F) * 2 - 1) ))
            y1 = _wrap_tensor(x,y1)

        return ME.cat(y1,y2)

class SparseBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SparseBottleneck, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=out_channels, out_channels=out_channels, kernel_size=1,dimension=3)
        self.conv3 = ME.MinkowskiConvolution(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,dimension=3)
        self.lrelu = ME.MinkowskiLeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights_me_xavier([self.conv1, self.conv2], 0.1)
        initialize_weights_me(self.conv3, 0)
        
    def forward(self, x, mask=None):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        if mask is not None:
            conv3 *= mask
        return conv3

class SparseSqueezeLayer(nn.Module):
    def __init__(self, in_channels, factor, squeeze_type):
        super().__init__()
        self.factor = factor
        self.squeeze_type=squeeze_type
        self.sparse_squeeze3d, self.sparse_unsqueeze3d = self.sparse_squeeze_layers(in_channels, factor)
        self.avg_pooling_layer = ME.MinkowskiAvgPooling(kernel_size=factor,stride=factor,dimension=3)
        for param in self.sparse_squeeze3d.parameters():
            param.requires_grad=False
        for param in self.sparse_unsqueeze3d.parameters():
            param.requires_grad=False

    def forward(self, input, reverse=False):
        if not reverse:
            if self.squeeze_type=="avg":    
                # Get a mask to put the mean of the tensors where it would produce 0s
                in_mask = _wrap_tensor(input,torch.ones_like(input.F))
                mask = self.sparse_squeeze3d(in_mask)
                mask._F=abs(mask.F-1)
                input_avg = self.avg_pooling_layer(input)
                input_avg._F=input_avg.F.repeat(1,self.factor**3)
                avg_mask = mask*input_avg
                output = self.sparse_squeeze3d(input)  # Squeeze in forward

                return output + avg_mask

            elif self.squeeze_type=="naive":
            
                output = self.sparse_squeeze3d(input)  # Squeeze in forward

                return output

            elif self.squeeze_type=="nn":
                
                raise NotImplementedError("The nearest neighbor mode has not been implemented")

        else:
            output = self.sparse_unsqueeze3d(input)
            return output
        
    def jacobian(self, x, reverse=False):
        return 0
        
    def sparse_squeeze_layers(self, in_channels, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input

        ch_in = in_channels
        ch_out = ch_in*(factor**3)
        
        # Create the squeeze layer based on a convolution with the correct weights
        sparse_squeeze=ME.MinkowskiConvolution(out_channels=ch_out,
                                               in_channels=ch_in,
                                               kernel_size=factor,
                                               stride=factor,
                                               dimension=3)
        # Make all the weights 0s to be able to sample the correct values
        sparse_squeeze.kernel.data=torch.zeros_like(sparse_squeeze.kernel.data)

        # Create the unsqueeze layer based on a convolution with the correct weights
        sparse_unsqueeze=ME.MinkowskiConvolutionTranspose(out_channels=ch_in,
                                                        in_channels=ch_out,
                                                        kernel_size=factor,
                                                        stride=factor,
                                                        dimension=3)
        # Make all the weights 0s to be able to sample the correct values
        sparse_unsqueeze.kernel.data=torch.zeros_like(sparse_unsqueeze.kernel.data)

        out_channel = np.arange(ch_out)
        in_channel = np.tile(np.arange(ch_in),ch_out//ch_in)
        x_index = np.tile(np.repeat([0,1],ch_out//2//2),2)
        y_index = np.tile(np.repeat([0,1],ch_out//2//2//2),2*2)
        z_index = np.tile(np.repeat([0,1],ch_out//2),1)

        for i in range(ch_out):
            index = x_index[i]+y_index[i]*2+z_index[i]*2*2
            sparse_squeeze.kernel.data[index,in_channel[i],out_channel[i]]=1
            sparse_unsqueeze.kernel.data[index,out_channel[i],in_channel[i]]=1

        return sparse_squeeze, sparse_unsqueeze

class SparseInvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, reverse):
        w_shape = self.w_shape
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1],1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1],1)
        return weight

    def forward(self, input, reverse=False):
        weight = self.get_weight(reverse)
        z=[]

        if not reverse:
            if len(input)==1:
                z = F.conv1d(torch.unsqueeze(torch.transpose(input.F,0,1),dim=0), weight)
                z_sparse = _wrap_tensor(input,torch.transpose(torch.squeeze(z,dim=0),0,1))
                return z_sparse

            total_length=0
            for perm, tensor in zip(input.decomposition_permutations, input.decomposed_features):
                tensor = tensor[self.invert_permutation_torch(perm-total_length)]
                z.append(F.conv1d(torch.unsqueeze(torch.transpose(tensor,0,1),dim=0), weight))
                total_length+=len(tensor)

            z = torch.cat(z,dim=2)
            z_sparse = _wrap_tensor(input,torch.transpose(torch.squeeze(z,dim=0),0,1))
            return z_sparse
        else:
            if len(input)==1:
                z = F.conv1d(torch.unsqueeze(torch.transpose(input.F,0,1),dim=0), weight)
                z_sparse = _wrap_tensor(input,torch.transpose(torch.squeeze(z,dim=0),0,1))
                return z_sparse
            
            total_length=0
            for perm, tensor in zip(input.decomposition_permutations, input.decomposed_features):
                tensor = tensor[self.invert_permutation_torch(perm-total_length)]
                z.append(F.conv1d(torch.unsqueeze(torch.transpose(tensor,0,1),dim=0), weight))
                total_length+=len(tensor)

            z = torch.cat(z,dim=2)
            z_sparse = _wrap_tensor(input, torch.transpose(torch.squeeze(z,dim=0),0,1))
            return z_sparse

    def invert_permutation_torch(self,permutation):
        inv = torch.empty_like(permutation).to(permutation.device)
        inv[permutation] = torch.arange(len(inv), dtype=inv.dtype, device =inv.device)
        return inv