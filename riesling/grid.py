import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import cmasher
import contextlib
plt.rcParams['font.size'] = 20
from .util import *

def get_grid(f, ifr, ic, comp, dset):
    fn = get_comp(comp)
    D = f[dset]
    grid = fn(D[:, :, :, ifr, ic])
    return grid

def planes(file, dset='cartesian', title=None, pos=None, ifr=0, ic=0, figsize=8, real=False):
    """3 plane plot of 3D image data in an h5 file

    Args:
        file  (str): Path to .h5 file
        dset  (str): Dataset within h5 file to plot. Default "image"
        title (str): Plot title. Defaults to ''
        pos   (int * 3): Slice indices
        iv    (int): Volume index
        ifr    (int): Echo/basis index
        comp  (str): mag/pha/real/imaginary. Default mag
        cmap  (str): colormap. Defaults to 'gray'
        clim  (float * 2): Window limits. Defaults to (2,98) percentiles
    """

    with h5py.File(file, 'r') as f:
        if real:
            img1 = get_grid(f, ifr, ic, 'real', dset)
            img2 = get_grid(f, ifr, ic, 'imag', dset)
            cm1 = 'cmr.iceburn'
            cm2 = 'cmr.iceburn'
        else:
            img1 = get_grid(f, ifr, ic, 'log', dset)
            img2 = get_grid(f, ifr, ic, 'pha', dset)
            cm1 = 'cmr.ember'
            cm2 = 'cmr.infinity'
    [nz, ny, nx] = img1.shape

    if not (pos):
        pos = (int(nz/2), int(ny/2), int(nx/2))

    fig, ax = plt.subplots(2, 3, figsize=(figsize*3, figsize), facecolor='black')
    ax[0, 0].imshow(get_slice(img1, pos[2], 'x'), cmap=cm1, interpolation='none')
    ax[0, 1].imshow(get_slice(img1, pos[1], 'y'), cmap=cm1, interpolation='none')
    mag_im = ax[0, 2].imshow(get_slice(img1, pos[0], 'z'), cmap=cm1, interpolation='none')
    ax[1, 0].imshow(get_slice(img2, pos[2], 'x'), cmap=cm2, interpolation='none')
    ax[1, 1].imshow(get_slice(img2, pos[1], 'y'), cmap=cm2, interpolation='none')
    pha_im = ax[1, 2].imshow(get_slice(img2, pos[0], 'z'), cmap=cm2, interpolation='none')
    # for a in ax:
    #     a.axis('image')
    fig.tight_layout(pad=0)

    cb = fig.colorbar(mag_im, location='right', ax=ax[0, :])
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    cb = fig.colorbar(pha_im, location='right', ax=ax[1, :])
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.suptitle(title, color='white')
    
    plt.close()
    return fig


def slices(file, dset='cartesian', title=None, ifr=0, ic=0,
           nslice=4, nrows=1, axis='z', start=None, stop=None,
           comp='real', cmap=None, clim=None, cbar=True, figsize=3):
    """Plot slices along one axis

    Args:
        file (str): .h5 file to load
        dset (str): Dataset within the .h5 file (default 'image')
        title (str): Plot title. Defaults to ''
        nslice (int): Number of slices
        nrows (int): Number of rows to plot slices over
        axis (str) : 'x' 'y' or 'z'
        ifr   (int): Echo index, default 0
        iv   (int): Volume to slice, default 0
        comp (str, opt): mag/pha/real/imaginary. Default mag
        cmap (str): colormap. Defaults to 'gray'.
        clim (float, float): Lower/upper window limits
    """
    with h5py.File(file, 'r') as f:
        img = get_grid(f, ifr, ic, comp, dset)
    [nz, ny, nx] = img.shape

    if not clim:
        clim = get_clim(img, comp)
    if not cmap:
        cmap = get_cmap(comp)

    if axis == 'z':
        maxn = nz - 1
    elif axis == 'y':
        maxn = ny - 1
    else:
        maxn = nx - 1
    if start is None:
        start = 0.25
    if stop is None:
        stop = 0.75
    slpos = np.floor(np.linspace(start*maxn, stop*maxn, nslice, endpoint=True)).astype(int)

    ncols = int(np.ceil(nslice / nrows))
    fig, all_ax = plt.subplots(nrows, ncols, figsize=(
        figsize*ncols, figsize*nrows), facecolor='black')

    for ir in range(nrows):
        if nrows > 1:
            if ncols > 1:
                sl_ax = all_ax[ir, :]
            else:
                sl_ax = all_ax[ir]
        else:
            sl_ax = all_ax
        for ic in range(ncols):
            sl = (ir * ncols) + ic
            if sl < nslice:
                if ncols > 1:
                    ax = sl_ax[ic]
                else:
                    ax = sl_ax
                data = get_slice(img, slpos[sl], axis)
                im = ax.imshow(data, cmap=cmap,
                               vmin=clim[0], vmax=clim[1])
                ax.axis('off')
    fig.tight_layout(pad=0)
    if cbar == True:
        add_colorbar(fig, im, all_ax, title, clim)
    else:
        if title is not None:
            fig.suptitle(title, color='white')
    plt.close()
    return fig

def diff(fnames, titles=None, dsets=['cartesian'],
         axis='z', slpos=None, ifr=0, ic=0,
         comp='log', clim=None, cmap='cmr.amber',
         difflim=None, diffmap='cmr.iceburn',
         figsize=4):
    """Plot the difference between grids

    Args:
        fnames (str): Paths to .h5 files
        titles (str): Titles
        dset (str): Dataset within h5 file to plot. Default "image"
        title (str): Plot title. Defaults to ''.
        sli  (int): Slice axis (0='x',1='y',2='z')
        iz   (int): Slice index for z-axis. Default 2.
        iv   (int): Volume to slice, default 0.
        cmap (str): colormap. Defaults to 'gray'.
        clim (float): Image window.
        difflim (float): Difference window (%)
        comp (str, opt): mag/pha/real/imaginary. Default mag
    """

    if len(dsets) == 1:
        dsets = dsets * len(fnames)

    try:
        iterator = iter(ifr)
    except:
        ifr = [ifr] * len(fnames)

    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fn, 'r')) for fn in fnames]
        imgs = [get_grid(f, ifr, ic, comp, ds) for f, ds in zip(files, dsets)]

        nI = len(imgs)

        if titles is not None:
            if len(titles) != len(fnames):
                warnings.warn('Number of titles and files did not match')
                return

        for ii in range(nI):
            for jj in range(nI - ii - 1):
                if (np.shape(imgs[ii]) != np.shape(imgs[ii+1+jj])):
                    warnings.warn('Image dimensions did not match')
                    return

        [nz, ny, nx] = np.shape(imgs[0])
        if not slpos:
            slpos = mid_slice(axis, nx, ny, nz)

        fn = get_comp(comp)
        slices = [get_slice(img, slpos, axis) for img in imgs]
        if not clim:
            clim = (np.inf, -np.inf)
            for sl in slices:
                temp_lim = np.nanpercentile(sl, (2, 98))
                clim = [np.amin([clim[0], temp_lim[0]]),
                        np.amax([clim[1], temp_lim[1]])]
            if clim[0] < 0:
                clim[1] = np.amax([np.abs(clim[0]), np.abs(clim[1])])
                clim[0] = -clim[1]

        diffs = []
        for ii in range(nI):
            diffs.append([])
            for jj in range(ii):
                diffs[ii].append((slices[ii] - slices[jj]) * 100 / clim[1])

        if not difflim:
            difflim = (np.inf, -np.inf)
            for ii in range(nI):
                for jj in range(ii):
                    templim = np.nanpercentile(diffs[ii][jj], (2, 98))
                    difflim = [np.amin([difflim[0], templim[0]]), np.amax(
                        [difflim[1], templim[1]])]
            if difflim[0] < 0:
                difflim[1] = np.amax([np.abs(difflim[0]), np.abs(difflim[1])])
                difflim[0] = -difflim[1]

        fig, ax = plt.subplots(nI, nI, figsize=(
            nI*figsize, nI*figsize), facecolor='black')
        for ii in range(nI):
            imi = ax[ii, ii].imshow(slices[ii], cmap=cmap,
                                    vmin=clim[0], vmax=clim[1])
            ax[ii, ii].axis('off')
            if titles is not None:
                ax[ii, ii].text(0.5, 0.9, titles[ii], color='white',
                                transform=ax[ii, ii].transAxes, ha='center')
            for jj in range(ii):
                imd = ax[jj, ii].imshow(diffs[ii][jj], cmap=diffmap,
                                        vmin=difflim[0], vmax=difflim[1])
                ax[jj, ii].axis('off')
            for jj in range(ii, nI):
                ax[jj, ii].set_facecolor('black')
                ax[jj, ii].axis('off')
        # axi = fig.add_axes([0.02, 0.2, 0.02, 0.6])
        fig.subplots_adjust(wspace=0, hspace=0)
        cbi = fig.colorbar(imi, ax=ax, location='left',
                           aspect=50, pad=0.01, shrink=0.8)
        cbi.ax.xaxis.set_tick_params(color='w', labelcolor='w')
        cbi.ax.yaxis.set_tick_params(color='w', labelcolor='w')
        # axd = fig.add_axes([0.98, 0.2, 0.02, 0.6])
        cbd = fig.colorbar(imd, ax=ax, location='right',
                           aspect=50, pad=0.01, shrink=0.8)
        cbd.ax.xaxis.set_tick_params(color='w', labelcolor='w')
        cbd.ax.yaxis.set_tick_params(color='w', labelcolor='w')

        plt.close()
        return fig
