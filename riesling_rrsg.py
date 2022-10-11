####################################################
#
# Convert the CG SENSE Reproducibility challenge brain
# dataset to riesling h5 format.
#
# How to use:
# - Download brain challenge and reference dataset
#       download_rrsg_data.sh
#
# - Create folder for riesling data named riesling_data
# Run script
#   python3 convert_data.py
#
# Emil Ljungberg and Tobias Wood
# December 2020, King's Colleg London
####################################################

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Data types as used in riesling
global DTYPE_ThreeD
global DTYPE_ThreeDStack
DTYPE_ThreeD = 1
DTYPE_ThreeDStack = 2


def create_info(matrix, channels, read_points, spokes, volumes, tr, voxel_size, origin, direction):
    D = np.dtype({'names': [
        'type',
        'matrix',
        'channels',
        'read_points',
        'spokes',
        'volumes',
        'frames',
        'tr',
        'voxel_size',
        'origin',
        'direction'
    ],
        'formats': [
        '<i8',
        ('<i8', (3,)),
        '<i8',
        '<i8',
        '<i8',
        '<i8',
        '<i8',
        '<f4',
        ('<f4', (3,)),
        ('<f4', (3,)),
        ('<f4', (9,))
    ]
    })

    info = np.array([(DTYPE_ThreeDStack, matrix, channels, read_points, spokes, volumes, 1,
                      tr, voxel_size, origin, direction)], dtype=D)

    return info


def convert_rrsg(input_fname, output_fname, matrix, voxel_size):
    data_f = h5py.File(input_fname, 'r')
    rawdata = data_f['rawdata'][...]
    traj = data_f['trajectory'][...]
    data_f.close()

    # Scale trajectory
    traj = traj/np.max(abs(traj)) * 0.5
    traj = traj.transpose((2, 1, 0))
    [npoints, nshots, nd] = np.shape(traj)

    # Strip 3rd dimension of radial dataset
    print(traj)
    if nd == 3:
        traj = traj[:,:,0:2]
    print(traj)

    # Create info struct
    read_points = np.shape(rawdata)[1]
    spokes = np.shape(rawdata)[2]
    channels = np.shape(rawdata)[3]
    volumes = 1
    tr = 1
    lo_scale = 1
    origin = [0, 0, 0]
    direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    h5_info = create_info(matrix, channels, read_points, spokes,
                          volumes, tr, voxel_size, origin, direction)
    # Reshape data
    rawdata = np.transpose(rawdata, [0, 2, 1, 3])

    # Create new H5 file
    out_f = h5py.File(output_fname, 'w')
    out_f.create_dataset("info", data=h5_info)
    out_f.create_dataset('trajectory', data=traj,
                         chunks=np.shape(traj), compression="gzip")
    out_f.create_dataset("noncartesian", dtype='c8', data=rawdata,
                         chunks=np.shape(rawdata), compression="gzip")
    out_f.close()

    print("H5 file saved to {}".format(output_fname))
