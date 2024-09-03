# MIT License

# Copyright (c) 2020 Maurice Quach

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Author: Maurice Quach 
# From: https://github.com/mauriceqch/pcc_geo_cnn_v2/blob/master/src/utils/mpeg_parsing.py
#
# Modified: Rodrigo Borba Pinheiro
# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import re

def parse_pcerror(path):
    with open(path, 'r') as f:
        s = f.read()
    
    d1_error = False
    d2_error = False
    att_error = False
    ########################################################################################

    try:
        pc_sizes = re.search(r'Point cloud sizes for org version, dec version, and the scaling ratio: (\d+), (\d+), (\d+)', s, re.MULTILINE)
        input_point_count = int(pc_sizes.group(1))
        decoded_point_count = int(pc_sizes.group(2))


    except AttributeError as e:
        print(s)
        raise e

    ########################################################################################
    ############################## parse the d1 psnr #######################################
    ########################################################################################
    try:
        # d1_mae = float(re.search(r'maeF      \(p2point\): (.+)', s, re.MULTILINE).group(1))
        d1_mse = float(re.search(r'mseF      \(p2point\): (.+)', s, re.MULTILINE).group(1))
        d1_psnr = float(re.search(r'mseF,PSNR \(p2point\): (.+)', s, re.MULTILINE).group(1))

    except AttributeError as e:
        d1_error = True

    ########################################################################################
    ############################## parse the d2 psnr #######################################
    ########################################################################################

    try:
        # d2_mae = float(re.search(r'maeF      \(p2point\): (.+)', s, re.MULTILINE).group(1))
        d2_mse = float(re.search(r'mseF      \(p2plane\): (.+)', s, re.MULTILINE).group(1))
        d2_psnr = float(re.search(r'mseF,PSNR \(p2plane\): (.+)', s, re.MULTILINE).group(1))

    except AttributeError as e:
        d2_error = True

    ########################################################################################
    ########################## parse the atttibutes psnr ###################################
    ########################################################################################
    try:
        y_mse = float(re.search(r'c\[0\],    F         : (.+)', s, re.MULTILINE).group(1))
        u_mse = float(re.search(r'c\[1\],    F         : (.+)', s, re.MULTILINE).group(1))
        v_mse = float(re.search(r'c\[2\],    F         : (.+)', s, re.MULTILINE).group(1))
        # y_mae = float(re.search(r'c\[0\],    maeF         : (.+)', s, re.MULTILINE).group(1))
        # u_mae = float(re.search(r'c\[1\],    maeF         : (.+)', s, re.MULTILINE).group(1))
        # v_mae = float(re.search(r'c\[2\],    maeF         : (.+)', s, re.MULTILINE).group(1))
        y_psnr = float(re.search(r'c\[0\],PSNRF         : (.+)', s, re.MULTILINE).group(1))
        u_psnr = float(re.search(r'c\[1\],PSNRF         : (.+)', s, re.MULTILINE).group(1))
        v_psnr = float(re.search(r'c\[2\],PSNRF         : (.+)', s, re.MULTILINE).group(1))
    except AttributeError:
        att_error = True
    
    if (att_error):
        return {
                "input_point_count": input_point_count,
                "decoded_point_count": decoded_point_count,
                # 'd1_mae': d1_mae,
                'd1_mse': d1_mse,
                'd1_psnr': d1_psnr,
                # 'd2_mae': d2_mae,
                'd2_mse': d2_mse,
                'd2_psnr': d2_psnr,
                }

    elif (d1_error and d2_error):
        return {
                "input_point_count": input_point_count,
                "decoded_point_count": decoded_point_count,
                'y_mse': y_mse,
                'u_mse': u_mse,
                'v_mse': v_mse,
                # 'y_mae': y_mae,
                # 'u_mae': u_mae,
                # 'v_mae': v_mae,
                'y_psnr': y_psnr,
                'u_psnr': u_psnr,
                'v_psnr': v_psnr
                }
    
    elif (d1_error):
        return {
                "input_point_count": input_point_count,
                "decoded_point_count": decoded_point_count,
                # 'd2_mae': d2_mae,
                'd2_mse': d2_mse,
                'd2_psnr': d2_psnr,
                'y_mse': y_mse,
                'u_mse': u_mse,
                'v_mse': v_mse,
                # 'y_mae': y_mae,
                # 'u_mae': u_mae,
                # 'v_mae': v_mae,
                'y_psnr': y_psnr,
                'u_psnr': u_psnr,
                'v_psnr': v_psnr
                }
    
    elif (d2_error):
        return {
                "input_point_count": input_point_count,
                "decoded_point_count": decoded_point_count,
                # 'd1_mae': d1_mae,
                'd1_mse': d1_mse,
                'd1_psnr': d1_psnr,
                'y_mse': y_mse,
                'u_mse': u_mse,
                'v_mse': v_mse,
                # 'y_mae': y_mae,
                # 'u_mae': u_mae,
                # 'v_mae': v_mae,
                'y_psnr': y_psnr,
                'u_psnr': u_psnr,
                'v_psnr': v_psnr
                }

    else: 
        return {
                "input_point_count": input_point_count,
                "decoded_point_count": decoded_point_count,
                # 'd1_mae': d1_mae,
                'd1_mse': d1_mse,
                'd1_psnr': d1_psnr,
                # 'd2_mae': d2_mae,
                'd2_mse': d2_mse,
                'd2_psnr': d2_psnr,
                'y_mse': y_mse,
                'u_mse': u_mse,
                'v_mse': v_mse,
                # 'y_mae': y_mae,
                # 'u_mae': u_mae,
                # 'v_mae': v_mae,
                'y_psnr': y_psnr,
                'u_psnr': u_psnr,
                'v_psnr': v_psnr
                }
