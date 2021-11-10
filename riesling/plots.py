import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import colorcet as cc
import warnings
plt.rcParams['font.size'] = 20

def plot_basis(path):
    f = h5py.File(path, 'r')
    basis = f['basis'][:]
    f.close()
    plt.plot(basis.T)
    plt.legend([str(x) for x in range(basis.shape[1])])
    plt.show()

def plot_dynamics(path):
    f = h5py.File(path, 'r')
    dyn = f['dynamics'][:]
    f.close()
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(dyn)
    plt.show()

def plot_trajectory(filename, skip_read=1, skip_spoke=1):
    with h5py.File(filename) as f:
        traj = np.array(f['trajectory'])
        print(traj.shape)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].grid()
        ax[0].scatter(traj[::skip_spoke, ::skip_read, 0],
                      traj[::skip_spoke, ::skip_read, 1], s=0.5)
        ax[0].set_aspect('equal')
        ax[1].grid()
        ax[1].scatter(traj[::skip_spoke, ::skip_read, 0],
                      traj[::skip_spoke, ::skip_read, 2], s=0.5)
        ax[1].set_aspect('equal')
        ax[2].grid()
        ax[2].scatter(traj[::skip_spoke, ::skip_read, 1],
                      traj[::skip_spoke, ::skip_read, 2], s=0.5)
        ax[2].set_aspect('equal')
        fig.tight_layout()
        plt.close()
    return fig
