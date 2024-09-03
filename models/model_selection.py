# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

def model_selection(args):
    """Compresses a Point Cloud.
    Inputs:
        args
            args.arch_type = the type of architecture to be used, either NF for normalizing flow or VAE for variational auto-encoder
            args.N = The number of filters N
            args.M = The number of filters M
            args.num_scales = The number of scales in the gaussian
            args.scale_min = The minimum scale 
            args.scale_max = The maximum scale
    Outputs:
        The model to be used in the current operation (train,inference)
  """

    if args.arch_type=="NF":
        from models.NF_PCAC import NF_PCAC
        model = NF_PCAC(args.M, 
                        args.N_levels, 
                        args.num_scales, 
                        args.scale_min, 
                        args.scale_max,
                        args.enh_channels,
                        args.attention_channels,
                        args.squeeze_type)

    # ADD ARBITRARY MODEL HERE IF NEEDED
    # elif args.arch_type=="new_model":
    #     from models.new_model import new_model
    #     model = new_model(...)

    else:
        print(args.arch_type)
        print("Not known architecture type, use NF")

    return model