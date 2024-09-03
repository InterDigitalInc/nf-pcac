# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

from plyfile import PlyData, PlyElement
import numpy as np

def read_PC(filename, YUV=False, out255=False):
    """Takes as inputs the string with the filename to read and outputs PC data
    Inputs:
        filename = string, path to the .ply file
        YUV = boolean, default is false, if set to true the Crgb output will be in the YUV color space
        out255 = boolean, default is false, if set to true the output will be in 0-255
    Outputs:
        points = np.array(shape=(N,6)) 
            To access coordinates -> points[:,:3]
            To access colors -> points[:,3:]
    """
    ply_raw = PlyData.read(filename)['vertex'].data
    points = np.vstack((ply_raw['x'], ply_raw['y'], ply_raw['z'],ply_raw["red"],ply_raw["green"],ply_raw["blue"])).transpose()
    points = np.ascontiguousarray(points)
    if YUV:
        points[:,3:] = RGB2YUV(points[:,3:])
    if not out255:
        points[:,3:] = points[:,3:]/255.
    
    return points

def write_PC(points, filename, inYUV=False, in255=False):
    """Takes as inputs the np.array with the points of the point cloud to write into a file.
    Inputs:
        points = np.array(shape=(N,6)) with coordinates and color
        filename = string, path to the .ply file
        inYUV = boolean, default is false, if set to true if the the input points is in the YUV color space
        in255 = boolean, default is false, if set to true, means the input data is in the range [0, 255]
    """
    #
    if not in255:
        points[:,3:]*=255.
    if inYUV:
        points[:,3:] = YUV2RGB(points[:,3:])

    vertex = list(zip(points[:,0], points[:,1], points[:,2],points[:,3],points[:,4],points[:,5]))
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elements = PlyElement.describe(vertex, "vertex")
    PlyData([elements]).write(filename)

def RGB2YUV(RGB):
    """Takes as inputs the np.array with the RGB colors and transform it to a YUV
    Inputs:
        RGB = np.array(shape=(N,3)) [0,255]
    Outputs:
        YUV = np.array(shape=(N,3))"""
    RGB1 = np.concatenate((RGB,np.ones((RGB.shape[0],1))), axis=1)

    Q = np.array([[0.2126, -0.1146, 0.5],
                  [0.7152, -0.3854, -0.4542],
                  [0.0722, 0.5, -0.0458],
                  [0, 127.5, 127.5]])
    
    YUV = np.matmul(RGB1,Q)
    return YUV

def YUV2RGB(YUV):
    """Takes as inputs the np.array with the YUV colors and transform it to a RGB
    Inputs:
        YUV = np.array(shape=(N,3)) [0,255]
    Outputs:
        RGB = np.array(shape=(N,3)) [0,255]
        """

    YUV[:,1:]-=127.5
    M = np.array([[1, 1, 1],
                  [0, -0.1873, 1.8556],
                  [1.5748, -0.4681, 0]])
    RGB = np.matmul(YUV,M)

    return RGB