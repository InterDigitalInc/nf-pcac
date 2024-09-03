# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import torch.nn as nn
import torch
import math
from compressai.entropy_models import EntropyBottleneck
from compressai.models.utils import update_registered_buffers

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=100):
        """ The rate-distortion loss calculation 
        Params:
            lmbda = float, the trade-off between rate and distortion (expected to be a number between 100 and infinite since MSE is computed in YUV space [0,1])
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        """ The rate-distortion loss calculation 
        Input:
            output = tensor, the output of the model after compression/decompression
            target = tensor, the ground truth point cloud
        Output:
            out = dictionary with the 3 different losses:
                out["mse_loss] = the mean squared error loss between output and target
                out["bpp_loss] = the log likehood of the values
                out["loss"] = the lambda*distortion + rate complete loss
        """
        out = {}

        
        num_voxels, _ = target.size()

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_voxels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = self.mse(output["x_hat"].F, target.F)

        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]
        
        return out

class CompressionModel(nn.Module):
    "Base class for the NF model with an entropy bottleneck"

    def __init__(self,entropy_bottleneck_channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m,EntropyBottleneck))
        return aux_loss

    def forward(self,*args):
        raise NotImplementedError()

    def update(self,force=False):
        updated = False
        for m in self.children():
            if not isinstance(m,EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated
    
    def load_state_dict(self,state_dict):

        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

