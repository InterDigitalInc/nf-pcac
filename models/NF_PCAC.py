# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import math
import torch
import torch.nn as nn
from layers.NF_layers import *
from models.hyperprior import HyperAnalysisTransform, HyperSynthesisTransform
from compressai.layers import *
from models.compress_model_base import CompressionModel
from compressai.models.utils import update_registered_buffers
from compressai.entropy_models import GaussianConditional

import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiFunctional import _wrap_tensor

class InvCore(nn.Module):
    def __init__(self, M, squeeze_type,  N_levels=3, dimensions=3):
        super(InvCore, self).__init__()
        # The number of channels in the input PC (RGB or YUV)
        self.in_nc = 3
        self.intermediate_channels = self.in_nc
        # The number of channels in the output of the channel average module
        self.out_nc = M

        self.operations = nn.ModuleList()

        for _ in range(N_levels):
            self.operations.append(SparseSqueezeLayer(self.intermediate_channels,2,squeeze_type))
            self.intermediate_channels *= 2**dimensions
            self.operations.append(SparseInvertibleConv1x1(self.intermediate_channels))
            self.operations.append(SparseCouplingLayer(self.intermediate_channels//4, 3*self.intermediate_channels//4, 3))
            self.operations.append(SparseCouplingLayer(self.intermediate_channels//4, 3*self.intermediate_channels//4, 3))

    def forward(self, x, reverse=False):
        if not reverse:
            for op in self.operations:
                x = op.forward(x, False)
            n, c = x.size()
            y = torch.mean(x.F.view(n, c//self.out_nc, self.out_nc), dim=1)
            y = _wrap_tensor(x,y)
        else:
            times = self.intermediate_channels // self.out_nc
            y = x.F.repeat(1, times)
            y = _wrap_tensor(x,y)
            
            for op in reversed(self.operations):
                y = op.forward(y, True)
        return y


class NF_PCAC(CompressionModel):

    def __init__(self, M=3*8*8,N_levels=3,num_scales=64,scale_min=.11,scale_max=256, enh=0, attention=0, squeeze_type="avg", **kwargs):
        super().__init__(entropy_bottleneck_channels=M)

        self.num_scales = num_scales
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.hyper_analysis_transform = HyperAnalysisTransform(conv_filters=[M,M,M,M,M,M],
                                          conv_kernel_size = [3,3,3,3,3],
                                          conv_strides = [1,1,2,1,2])
        self.hyper_synthesis_transform = HyperSynthesisTransform(conv_filters=[M,M,M,2*M,2*M,2*M],
                                           conv_kernel_size = [3,3,3,3,3],
                                           conv_strides = [2,1,2,1,1])
        if not enh==0:
            self.enh = SparseEnhModule(enh)
        else:
            self.enh = None

        self.inv = InvCore(M,squeeze_type=squeeze_type,N_levels=N_levels)

        if not attention==0:
            self.attention = SparseAttModule(attention)
        else:
            self.attention = None

        self.gaussian_conditional = GaussianConditional(None)
        

    def analysis_transform(self, x):
        
        if self.enh is not None:
            x = self.enh(x)
        
        x = self.inv(x)
        
        if self.attention is not None:
            x = self.attention(x)

        return x

    def synthesis_transform(self, x):
        
        if self.attention is not None:
            x = self.attention(x, reverse = True)
        
        x = self.inv(x, reverse=True)
        
        if self.enh is not None:
            x = self.enh(x, reverse=True)
        
        return x

    def forward(self, x):
        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)
        
        # Needs to unsqueeze and transpose to use compressAI bottleneck (fully factorized model)
        z_hat, side_bits = self.entropy_bottleneck(torch.unsqueeze(torch.transpose(z.F,0,1),dim=0))

        # Put the features back in the sparse tensor
        z_hat_sparse = _wrap_tensor(z, torch.transpose(torch.squeeze(z_hat,dim=0),0,1))

        gaussian_params = self.hyper_synthesis_transform(z_hat_sparse)

        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
        
        # Encode the latent with the hyperprior information decoded (mean and scale)
        y_hat, bits = self.gaussian_conditional(torch.unsqueeze(torch.transpose(y.F,0,1),dim=0), 
                                                torch.unsqueeze(torch.transpose(scales_hat,0,1),dim=0), 
                                                means=torch.unsqueeze(torch.transpose(means_hat,0,1),dim=0))
        
        # Put the features back in the sparse tensor
        y_hat_sparse = _wrap_tensor(y,torch.transpose(torch.squeeze(y_hat,dim=0),0,1))

        # Pass it through the NF backwards
        x_hat = self.synthesis_transform(y_hat_sparse)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": bits, "z": side_bits}
        }

    def compress(self, x, debug=False):

        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)

        z_strings = self.entropy_bottleneck.compress(torch.unsqueeze(torch.transpose(z.F,0,1),dim=0))
        z_hat = self.entropy_bottleneck.decompress(z_strings, [z.size()[0]])

        z_hat_sparse = _wrap_tensor(z,torch.transpose(torch.squeeze(z_hat,dim=0),0,1))

        gaussian_params = self.hyper_synthesis_transform(z_hat_sparse)
        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_strings = self.gaussian_conditional.compress(torch.unsqueeze(torch.transpose(y.F,0,1),dim=0),
                                                        torch.unsqueeze(torch.transpose(indexes,0,1),dim=0),
                                                        means=torch.unsqueeze(torch.transpose(means_hat,0,1),dim=0))

        if not debug:
            return {"strings": [y_strings, z_strings], "shape": [z.size()[0]]}

        else:
            return {"strings": [y_strings, z_strings], 
                    "shape": [z.size()[0]],
                    "y" : y,
                    "z" : z,
                    "z_hat_sparse": z_hat_sparse,
                    "gaussian_params": gaussian_params,
                    "indexes": indexes,
                    }


    def decompress(self, strings, latent_coords, hyper_latent_coords):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], [hyper_latent_coords.size()[0]])

        z_hat_sparse = _wrap_tensor(hyper_latent_coords, torch.transpose(torch.squeeze(z_hat,dim=0),0,1))

        gaussian_params = self.hyper_synthesis_transform(z_hat_sparse)
        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_hat = self.gaussian_conditional.decompress(strings[0],
                                                    torch.unsqueeze(torch.transpose(indexes,0,1),dim=0),
                                                    means=torch.unsqueeze(torch.transpose(means_hat,0,1),dim=0))

        y_hat_sparse = _wrap_tensor(latent_coords, torch.transpose(torch.squeeze(y_hat,dim=0),0,1))

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = self.synthesis_transform(y_hat_sparse)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def get_scale_table(self):
        return torch.exp(torch.linspace(math.log(self.scale_min), math.log(self.scale_max), self.num_scales))

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = self.get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated