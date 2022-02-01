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
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(basis.T)
    ax.legend([str(x) for x in range(basis.shape[1])])
    plt.show()


def dynamics(path):
    f = h5py.File(path, 'r')
    dyn = f['dynamics'][:]
    f.close()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dyn)
    plt.show()


def traj2d(filename, sl_read=slice(0, -1, 1), sl_spoke=slice(0, -1, 1)):
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


def traj3d(filename, sl_read=slice(0, -1, 1), sl_spoke=slice(0, -1, 1)):
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


def kspace(filename, dset='noncartesian', title=None, vol=0, coil=0,
           sl_read=slice(None, None, 1), sl_spoke=slice(None, None, 1),
           comp='mag'):
    with h5py.File(filename) as f:
        dsetw = f[dset]
        if dsetw.ndim == 3:
            data = np.array(f[dset][sl_spoke, sl_read, coil]).squeeze().T
        else:
            data = np.array(f[dset][vol, sl_spoke, sl_read, coil]).squeeze().T
        dmax = np.max(np.abs(data))
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), facecolor='w')
        if comp == 'mag':
            im0 = ax[0].imshow(np.log(np.abs(data+1E-10)), vmin=np.log(1E-10), vmax=np.log(dmax),
                               interpolation='nearest', cmap='cet_bmy')
            im1 = ax[1].imshow(np.angle(data), interpolation='nearest',
                               cmap='cet_colorwheel', vmin=-np.pi, vmax=np.pi)
        else:
            im0 = ax[0].imshow(np.real(data), interpolation='nearest', cmap='cet_coolwarm',
                               vmin=-dmax, vmax=dmax)
            im1 = ax[1].imshow(np.imag(data), interpolation='nearest', cmap='cet_coolwarm',
                               vmin=-dmax, vmax=dmax)
        fig.colorbar(im0, ax=ax[0], location='right')
        fig.colorbar(im1, ax=ax[1], location='right')
        ax[1].set_xlabel('Spoke')
        ax[0].set_ylabel('Readout')
        ax[1].set_ylabel('Readout')
        ax[0].axis('auto')
        ax[1].axis('auto')
        fig.tight_layout(pad=0)
        fig.suptitle(title)
        plt.close()
    return fig


def sdc(filename, dset='sdc', sl_read=slice(0, -1, 1), sl_spoke=slice(None, None, 1)):
    with h5py.File(filename) as f:
        data = np.array(f[dset][sl_spoke, sl_read]).T
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        im = ax.imshow(data, interpolation='nearest')
        ax.set_xlabel('Spoke')
        ax.set_ylabel('Readout')
        ax.axis('auto')
        ax.axis('auto')
        fig.colorbar(im, location='right')
        fig.tight_layout()
        plt.close()
    return fig
