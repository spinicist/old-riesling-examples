import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import colorcet as cc
import warnings
plt.rcParams['font.size'] = 20

def basis(path):
    f = h5py.File(path, 'r')
    basis = f['basis'][:]
    f.close()
    fig, ax = plt.subplots(figsize=(16,6))
    ax.plot(basis.T)
    ax.legend([str(x) for x in range(basis.shape[1])])
    plt.show()

def dynamics(path):
    f = h5py.File(path, 'r')
    dyn = f['dynamics'][:]
    f.close()
    fig, ax = plt.subplots(figsize=(16,6))
    ax.plot(dyn)
    plt.show()

def traj2d(filename, sl_read=slice(0,-1,1), sl_spoke=slice(0,-1,1)):
    with h5py.File(filename) as f:
        traj = np.array(f['trajectory'])
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].grid()
        ax[0].scatter(traj[sl_spoke, sl_read, 0],
                      traj[sl_spoke, sl_read, 1], s=0.5)
        ax[0].set_aspect('equal')
        ax[1].grid()
        ax[1].scatter(traj[sl_spoke, sl_read, 0],
                      traj[sl_spoke, sl_read, 2], s=0.5)
        ax[1].set_aspect('equal')
        ax[2].grid()
        ax[2].scatter(traj[sl_spoke, sl_read, 1],
                      traj[sl_spoke, sl_read, 2], s=0.5)
        ax[2].set_aspect('equal')
        fig.tight_layout()
        plt.close()
    return fig

def traj3d(filename, sl_read=slice(0,-1,1), sl_spoke=slice(0,-1,1)):
    with h5py.File(filename) as ff:
        traj = np.array(ff['trajectory'])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(traj[sl_spoke, sl_read, 0],
                    traj[sl_spoke, sl_read, 1],
                    traj[sl_spoke, sl_read, 2])
        fig.tight_layout()
        plt.close()
    return fig

def kspace(filename, dset='noncartesian', vol=0, coil=0, sl_read=slice(0,-1,1), sl_spoke=slice(0,512,1)):
    with h5py.File(filename) as f:
        data = np.array(f['noncartesian'][vol,sl_spoke,sl_read,coil]).squeeze().T
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        ax[0].imshow(np.log(np.abs(data+1E-10)))
        ax[1].imshow(np.angle(data))
        ax[1].set_xlabel('Spoke')
        ax[0].set_ylabel('Readout')
        ax[1].set_ylabel('Readout')
        ax[0].axis('auto')
        ax[1].axis('auto')
        fig.tight_layout()
        plt.close()
    return fig
