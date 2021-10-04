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


def create_info(matrix, voxel_size, read_points, read_gap, spokes_hi, spokes_lo, lo_scale,
                channels, volumes, tr, origin, direction):
    D = np.dtype({'names': [
        'matrix',
        'voxel_size',
        'read_points',
        'read_gap',
        'spokes_hi',
        'spokes_lo',
        'lo_scale',
        'channels',
        'volumes',
        'tr',
        'origin',
        'direction',
        'type'],
        'formats': [
        ('<i8', (3,)),
        ('<f4', (3,)),
        '<i8',
        '<i8',
        '<i8',
        '<i8',
        '<f4',
        '<i8',
        '<i8',
        '<f4',
        ('<f4', (3,)),
        ('<f4', (9,)),
        '<i8']
    })

    info = np.array([(matrix, voxel_size, read_points, read_gap, spokes_hi, spokes_lo, lo_scale,
                      channels, volumes, tr, origin, direction, DTYPE_ThreeDStack)], dtype=D)

    return info


def convert_data(input_fname, output_fname, matrix, voxel_size):
    data_f = h5py.File(input_fname, 'r')
    rawdata = data_f['rawdata'][...]
    traj = data_f['trajectory'][...]
    data_f.close()

    # Scale trajectory
    traj = traj/np.max(abs(traj)) * 0.5
    traj = traj.transpose((2, 1, 0))
    print(np.shape(traj))
    [npoints, nshots, nd] = np.shape(traj)

    # If 2D trajectory extend to third dimension
    if nd == 2:
        traj = np.pad(traj, ((0, 0), (0, 0), (0, 1)))

    # Create info struct
    read_points = np.shape(rawdata)[1]
    read_gap = 0
    spokes_hi = np.shape(rawdata)[2]
    spokes_lo = 0
    channels = np.shape(rawdata)[3]
    volumes = 1
    tr = 1
    lo_scale = 8
    origin = [0, 0, 0]
    direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    h5_info = create_info(matrix, voxel_size, read_points, read_gap, spokes_hi, spokes_lo, lo_scale,
                          channels, volumes, tr, origin, direction)
    # Reshape data
    print(rawdata.shape)
    rawdata = np.transpose(rawdata, [0, 2, 1, 3])
    print(rawdata.shape)

    # Create new H5 file
    out_f = h5py.File(output_fname, 'w')
    out_f.create_dataset("info", data=h5_info)
    out_f.create_dataset('trajectory', data=traj,
                         chunks=np.shape(traj), compression="gzip")
    out_f.create_dataset("noncartesian", dtype='c8', data=rawdata,
                         chunks=np.shape(rawdata), compression="gzip")
    out_f.close()

    print("H5 file saved to {}".format(output_fname))


def main():
    # Check if riesling folder exists
    if not os.path.isdir('riesling_data'):
        print("Directory riesling_data does not exists. Creating it")
        os.mkdir('riesling_data')

    # Challenge data
    f = 'rawdata_brain_radial_96proj_12ch.h5'
    convert_data(input_fname='rrsg_data/%s' % f,
                 output_fname='riesling_data/riesling_%s' % f,
                 matrix=[256, 256, 1], voxel_size=[0.78, 0.78, 2])

    # Reference data
    f = 'rawdata_spiral_ETH.h5'
    convert_data(input_fname='rrsg_data/%s' % f,
                 output_fname='riesling_data/riesling_%s' % f,
                 matrix=[220, 220, 1], voxel_size=[1, 1, 2])


if __name__ == "__main__":
    main()
