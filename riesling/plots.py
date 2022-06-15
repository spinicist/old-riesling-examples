import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import cmasher
import warnings
plt.rcParams['font.size'] = 20


def basis(path, sl_spoke=slice(None), b=slice(None)):
    with h5py.File(path, 'r') as f:
        basis = f['basis'][b,sl_spoke]
        print(basis.shape)
        f.close()
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(basis.T)
        ax.legend([str(x) for x in range(basis.shape[1])])
        plt.close()
        return fig


def dynamics(filename):
    with h5py.File(filename) as f:
        dyn = f['dynamics'][:]
        f.close()
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(dyn)
        plt.close()
        return fig


def traj2d(filename, sl_read=slice(None), sl_spoke=slice(None), color='read', sps=None):
    with h5py.File(filename) as f:
        traj = np.array(f['trajectory'])
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), facecolor='w')
        if color == 'read':
            c = np.tile(np.arange(len(traj[0, sl_read, 0])), len(traj[sl_spoke, 0, 0]))
        elif color == 'seg':
            c = np.tile(np.repeat(np.arange(sps),
                                  len(traj[0, sl_read, 0])),
                        int(len(traj[sl_spoke, 0, 0])/sps))
        else:
            c = np.tile(np.arange(len(traj[sl_spoke, 0, 0])), (len(traj[0, sl_read, 0]), 1)).ravel(order='F')
        ax[0].grid()
        ax[0].scatter(traj[sl_spoke, sl_read, 0],
                      traj[sl_spoke, sl_read, 1], c=c, s=0.5)
        ax[1].grid()
        ax[1].scatter(traj[sl_spoke, sl_read, 0],
                      traj[sl_spoke, sl_read, 2], c=c, s=0.5)
        ax[2].grid()
        ax[2].scatter(traj[sl_spoke, sl_read, 1],
                      traj[sl_spoke, sl_read, 2], c=c, s=0.5)

        if (np.max(np.abs(traj[:])) > 0.5):
            ax[0].set_aspect('equal') # Only set the plot to be square for stack-of-stars 
        else:
            [a.set_aspect('equal') for a in ax]
        fig.tight_layout()
        plt.close()
    return fig


def traj3d(filename, sl_read=slice(None), sl_spoke=slice(None), color='read', sps=None):
    with h5py.File(filename) as ff:
        traj = np.array(ff['trajectory'])
        print(traj.shape)
        if color == 'read':
            c = np.tile(np.arange(len(traj[0, sl_read, 0])), len(traj[sl_spoke, 0, 0]))
        elif color == 'seg':
            c = np.tile(np.repeat(np.arange(sps),
                                  len(traj[0, sl_read, 0])),
                        int(len(traj[sl_spoke, 0, 0])/sps))
        else:
            c = np.tile(np.arange(len(traj[sl_spoke, 0, 0])), (len(traj[0, sl_read, 0]), 1)).ravel(order='F')
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(traj[sl_spoke, sl_read, 0],
                   traj[sl_spoke, sl_read, 1],
                   traj[sl_spoke, sl_read, 2],
                   c=c, s=3)
        fig.tight_layout()
        plt.close()
    return fig


def kspace(filename, dset='noncartesian', title=None, vol=0, channel=0,
           sl_read=slice(None, None, 1), sl_spoke=slice(None, None, 1),
           comp='mag', clim=None):
    with h5py.File(filename) as f:
        dsetw = f[dset]
        if dsetw.ndim == 3:
            data = np.array(f[dset][sl_spoke, sl_read, channel]).squeeze().T
        else:
            data = np.array(
                f[dset][vol, sl_spoke, sl_read, channel]).squeeze().T
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), facecolor='w')
        if comp == 'mag':
            if clim is None:
                dmax = np.max(np.abs(data))
                clim = (np.log(1E-10), np.log(dmax))
            im0 = ax[0].imshow(np.log(np.abs(data+1E-10)), vmin=clim[0], vmax=clim[1],
                               interpolation='nearest', cmap='cmr.ember')
            im1 = ax[1].imshow(np.angle(data), interpolation='nearest',
                               cmap='cmr.infinity', vmin=-np.pi, vmax=np.pi)
        else:
            if clim is None:
                temp_lim = np.nanpercentile(np.abs(data), (98))
                clim = (-temp_lim, temp_lim)
            im0 = ax[0].imshow(np.real(data), interpolation='nearest', cmap='cmr.iceburn',
                               vmin=clim[0], vmax=clim[1])
            im1 = ax[1].imshow(np.imag(data), interpolation='nearest', cmap='cmr.iceburn',
                               vmin=clim[0], vmax=clim[1])
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
        fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w')
        im = ax.imshow(data, interpolation='nearest',
                       cmap='cmr.ember')
        ax.set_xlabel('Spoke')
        ax.set_ylabel('Readout')
        ax.axis('auto')
        ax.axis('auto')
        fig.colorbar(im, location='right')
        fig.tight_layout()
        plt.close()
    return fig
