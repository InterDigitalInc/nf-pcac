# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

from typing import Any
from torch.utils.data import Dataset
import numpy as np
from our_utils.transform_io_utils import read_PC
import MinkowskiEngine as ME

class CustomDataset(Dataset):
    def __init__(self, pcFileList:list=[], YUV=False, out255=False, data_augmentation=None):
        """
        Custom code for loading point clouds into a torch dataset
        Inputs: 
            pcFileList = list with all the .ply paths of PCs that are part of the dataset
            YUV = boolean if true, transform it to YUV space
            out255 = boolean if false will return the attributes in [0,1]
            data_augmentation = Data augmentation composed function
        Outputs (when called by a DataLoader or indexed):
            discrete_coords = Tensor with the coordinates
            unique_feats = Tensor with the respective features
        """
        self.YUV = YUV
        self.out255 = out255
        self.data_augmentation = data_augmentation

        assert (len(pcFileList)!=0)

        self.pcFileList = pcFileList

    def __len__(self):
        return len(self.pcFileList)
    
    def __getitem__(self, index) -> Any:

        points = read_PC(self.pcFileList[index],YUV=self.YUV)
        # Get the vertex and the colors
        V = points[:,:3]
        Colors = points[:,3:]

        # Apply data augmentation as needed
        if self.data_augmentation:
            V, Colors = self.data_augmentation(V,Colors)

        # Make sure it is quantized to be ingested in the framework
        discrete_coords, unique_feats = ME.utils.sparse_quantize(V.astype("float32"),
                                                                 Colors.astype("float32"),
                                                                 quantization_size=1)
        
        # If the PC has only one point (to avoid a bug)
        if len(Colors)==1:
            unique_feats=Colors.astype("float32")
                
        return discrete_coords, unique_feats.astype("float32")
    
