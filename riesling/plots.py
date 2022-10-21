import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import cmasher
import warnings
plt.rcParams['font.size'] = 20


def basis(path, sl_spoke=slice(None), b=slice(None)):
    with h5py.File(path, 'r') as f:
        basis = f['basis'][b,sl_spoke]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(basis.T)
        ax.legend([str(x) for x in range(basis.shape[1])])
        plt.close()
        return fig


def dynamics(filename, sl=slice(None)):
    with h5py.File(filename) as f:
        dyn = f['dynamics'][sl,:]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(dyn.T)
        ax.axhline(0)
        plt.close()
        return fig

def dictionary(filename):
    with h5py.File(filename) as f:
        d = f['dictionary']
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs = d[:, 0], ys = d[:, 1], zs = d[:, 2])
        plt.close()
        return fig

def traj2d(filename, sl_read=slice(None), sl_spoke=slice(None), color='read', sps=None):
    with h5py.File(filename) as f:
        traj = np.array(f['trajectory'])
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), facecolor='w')
        if color == 'read':
            c = np.tile(np.arange(len(traj[0, sl_read, 0])), len(traj[sl_spoke, 0, 0]))
        elif color == 'seg':
            c = np.tile(np.repeat(np.arange(sps),
                                  len(traj[0, sl_read, 0])),
                        int(len(traj[sl_spoke, 0, 0])/sps))
        else:
            c = np.tile(np.arange(len(traj[sl_spoke, 0, 0])), (len(traj[0, sl_read, 0]), 1)).ravel(order='F')
        ax.grid()
        ax.scatter(traj[sl_spoke, sl_read, 0],
                      traj[sl_spoke, sl_read, 1], c=c, s=0.5)
        ax.set_aspect('equal')
        fig.tight_layout()
        plt.close()
    return fig


def traj3d(filename, sl_read=slice(None), sl_spoke=slice(None), color='read', sps=None, angles=[30,-60,0]):
    with h5py.File(filename) as ff:
        traj = ff['trajectory'][sl_spoke, sl_read, :]
        if color == 'read':
            c = np.tile(np.arange(traj.shape[1], traj.shape[0]))
        elif color == 'seg':
            c = np.tile(np.repeat(np.arange(sps), traj.shape[1]), int(traj.shape[0]/sps))
        else:
            c = np.tile(np.arange(traj.shape[0]), (traj.shape[1], 1))
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')

        x, y, z = np.array([[-0.5,0,0],[0,-0.5,0],[0,0,-0.5]])
        u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
        ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1)
        ax.scatter(traj[:, :, 0], traj[:, :, 1], traj[:, :, 2],
                   c=c, s=3, cmap='cmr.lavender')

        ax.view_init(elev=angles[0], azim=angles[1], vertical_axis='z')
        fig.tight_layout()
        plt.close()
    return fig

def trajEnds(filename, sl_read=slice(None), sl_spoke=slice(None), color='read', sps=None, angles=[30,-60,0]):
    with h5py.File(filename) as ff:
        traj = np.array(ff['trajectory'])
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

        x, y, z = np.array([[-0.5,0,0],[0,-0.5,0],[0,0,-0.5]])
        u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
        ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1)
        print((traj[sl_spoke, sl_read, 0]).shape)
        ax.plot(xs=np.ravel(traj[sl_spoke, sl_read, 0], order='F'),
                ys=np.ravel(traj[sl_spoke, sl_read, 1], order='F'),
                zs=np.ravel(traj[sl_spoke, sl_read, 2], order='F'),
                c=c, cmap='cmr.lavender')
        # ax.set_axis_off()
        ax.grid('off')

        ax.view_init(elev=angles[0], azim=angles[1], vertical_axis='z')
        fig.tight_layout()
        plt.close()
    return fig

def kspace(filename, dset='noncartesian', title=None, vol=0, slc=0, channel=0,
           sl_read=slice(None, None, 1), sl_spoke=slice(None, None, 1),
           split=False, clim=None):
    with h5py.File(filename) as f:
        dsetw = f[dset]
        if dsetw.ndim == 4:
            data = np.array(f[dset][slc, sl_spoke, sl_read, channel]).squeeze().T
        else:
            data = np.array(
                f[dset][vol, slc, sl_spoke, sl_read, channel]).squeeze().T

        if split:
            if clim is None:
                temp_lim = np.nanpercentile(np.abs(data), (98))
                clim = (-temp_lim, temp_lim)
            fig, ax = plt.subplots(2, 1, figsize=(18, 12), facecolor='w')
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
        else:
            norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
            smap = cm.ScalarMappable(norm=norm, cmap='cmr.infinity')
            pha = np.angle(data)
            mag = np.log1p(np.abs(data))
            lims = np.nanpercentile(mag, (10, 90))
            mag = np.clip((mag - lims[0]) / (lims[1] - lims[0]), 0, 1)
            colorized = smap.to_rgba(pha, alpha=1., bytes=False)[:, :, 0:3]
            colorized = colorized * mag[:, :, None]
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w')
            ax.imshow(colorized)
            ax.set_xlabel('Spoke')
            ax.set_ylabel('Readout')
            ax.axis('auto')

        fig.tight_layout(pad=0)
        fig.suptitle(title)
        plt.close()
    return fig


def sdc(filename, dset='sdc', sl_read=slice(None, None, 1), sl_spoke=slice(None, None, 1), log=False, clim=None):
    with h5py.File(filename) as f:
        data = np.array(f[dset][sl_spoke, sl_read]).T
        if log:
            data = np.log(data)
            if clim is None:
                clim = (np.log(1E-10), np.max(data))
        elif clim is None:
            clim = np.nanpercentile(np.abs(data), (2, 98))
        fig, ax = plt.subplots(1, 1, figsize=(18, 6), facecolor='w')
        im = ax.imshow(data, interpolation='nearest',
                       cmap='cmr.ember', vmin=clim[0], vmax=clim[1])
        ax.set_xlabel('Spoke')
        ax.set_ylabel('Readout')
        ax.axis('auto')
        fig.colorbar(im, location='right')
        fig.tight_layout()
        plt.close()
    return fig
