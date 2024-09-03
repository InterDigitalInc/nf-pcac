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

# Modified: Rodrigo Borba Pinheiro
# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import numpy as np

def to_bytes(x, dtype):
    iinfo = np.iinfo(dtype)
    x = np.array(x, dtype=dtype)
    assert np.all(x <= iinfo.max), f'Overflow {x} {iinfo}'
    assert np.all(iinfo.min <= x), f'Underflow {x} {iinfo}'
    return x.tobytes()

def scalar_to_bytes(x, dtype):
    return to_bytes([x], dtype)

def read_from_buffer(f, n, dtype):
    return np.frombuffer(f.read(int(np.dtype(dtype).itemsize * n)), dtype=dtype)

def save_compressed_file(strings):
    """Saves the attributes of a point cloud and the hyperprior an unified bitstream"""
    ret = b''
    for string in strings:
        n_bytes_b = scalar_to_bytes(len(string[0]), np.uint32)
        ret += n_bytes_b + string[0]
    return ret

def load_compressed_file(f):
    """Loads the unified bitstream from an encoded point cloud"""
    strings = []
    for _ in range(2):
        n_bytes = read_from_buffer(f, 1, np.uint32)[0]
        string = f.read(int(n_bytes))
        strings.append([string])
    
    file_end = f.read()
    assert file_end == b'', f'File not read completely file_end {file_end}'

    return strings