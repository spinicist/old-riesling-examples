import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import cmasher
import contextlib
plt.rcParams['font.size'] = 20
from .util import *

def get_img(f, ifr, iv, ic, comp, dset):
    fn = get_comp(comp)
    D = f[dset]
    if D.ndim == 3:
        img = fn(D)
    elif D.ndim == 4:
        img = fn(D[:, :, :, ifr])
    elif D.ndim == 5:
        img = fn(D[iv, :, :, :, ifr])
    elif D.ndim == 6:
        img = fn(D[iv, :, :, :, ifr, ic])
    return img

def planes(file, dset='image', title=None, pos=None, iv=0, ifr=0, ic=0, figsize=5,
           comp='mag', cbar=True, cmap=None, clim=None):
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
        img = get_img(f, ifr, iv, ic, comp, dset)
    [nz, ny, nx] = img.shape

    if not (pos):
        pos = (int(nz/2), int(ny/2), int(nx/2))
    if not clim:
        clim = get_clim(img, comp)
    if not cmap:
        cmap = get_cmap(comp)

    fig, ax = plt.subplots(1, 3, figsize=(figsize*3, figsize), facecolor='black')
    ax[0].imshow(get_slice(img, pos[2], 'x'),
                 cmap=cmap, vmin=clim[0], vmax=clim[1], interpolation='bilinear')
    ax[0].axis('image')
    ax[1].imshow(get_slice(img, pos[1], 'y'),
                 cmap=cmap, vmin=clim[0], vmax=clim[1], interpolation='bilinear')
    ax[1].axis('image')
    im = ax[2].imshow(get_slice(img, pos[0], 'z'),
                      cmap=cmap, vmin=clim[0], vmax=clim[1], interpolation='bilinear')
    ax[2].axis('image')
    fig.tight_layout(pad=0)
    if cbar:
        cb = fig.colorbar(im, location='right', ax=ax)
        cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.suptitle(title, color='white')
    
    plt.close()
    return fig


def slices(file, dset='image', title=None, ifr=0, iv=0, ic=0,
           nslice=4, nrows=1, axis='z', start=None, stop=None,
           comp='mag', cmap=None, clim=None, cbar=True, figsize=3):
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
        img = get_img(f, ifr, iv, ic, comp, dset)
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


def series(file, dset='image', title=None, ifr=None, iv=None, ic=None,
           slc=slice(None), nrows=1, axis='z', slpos=0.5, figsize=3,
           comp='mag', cmap='gray', clim=None, cbar=True):
    """One slice through echoes or volumes

    Args:
        file (str): .h5 file to load
        dset (str): Dataset within the .h5 file (default 'image')
        title (str): Plot title. Defaults to ''
        ifr (int): Index into frames dim
        iv (int): Inde into volumes dim
        ic (int): Index into channel dim
        nrows (int): Number of rows to plot slices over
        axis (str) : 'x' 'y' or 'z'
        slpos (int): Which slice
        comp (str, opt): mag/pha/real/imaginary. Default mag
        cmap (str): colormap. Defaults to 'gray'.
        clim (float, float): Lower/upper window limits
    """

    fn = get_comp(comp)
    with h5py.File(file, 'r') as f:
        D = f[dset]
        if D.ndim == 4:
            # Channels only image, e.g. phantom
            [nz, ny, nx, nc] = D.shape
            img = fn(D).transpose(3, 0, 1, 2)
        elif D.ndim == 5:
            [nv, nz, ny, nx, nfr] = D.shape
            if iv is not None:
                img = fn(D[iv, :, :, :, slc].transpose((3, 0, 1, 2)))
            elif ifr is not None:
                img = fn(D[slc, :, :, :, ifr])
            else:
                raise Exception('Either ifr or iv must be specified')
        else:
            # Assume 6D
            [nv, nz, ny, nx, nfr, nc] = D.shape
            if iv is not None and ic is not None:
                img = fn(D[iv, :, :, :, slc, ic].transpose((3, 0, 1, 2)))
            elif iv is not None and ifr is not None:
                img = fn(D[iv, :, :, :, ifr, slc].transpose((3, 0, 1, 2)))
            elif ifr is not None and ic is not None:
                img = fn(D[slc, :, :, :, ifr, ic])
            else:
                raise Exception('A pair of ifr/ic/iv must be specified')
    nslice = img.shape[0]

    slpos = get_slpos(axis, nx, ny, nz, slpos)

    if not clim:
        clim = np.nanpercentile(img, (2, 98))
        if clim[0] < 0:
            clim[1] = np.max(np.abs(clim))
            clim[0] = -clim[1]

    ncols = int(np.ceil(nslice / nrows))
    fig, all_ax = plt.subplots(nrows, ncols, figsize=(
        figsize*ncols, figsize*nrows), facecolor='black')

    for ir in range(nrows):
        if nrows > 1:
            sl_ax = all_ax[ir, :]
        else:
            sl_ax = all_ax
        for ic in range(ncols):
            sl = (ir * ncols) + ic
            if sl < nslice:
                if hasattr(sl_ax, "__len__"):
                    ax = sl_ax[ic]
                else:
                    ax = sl_ax
                data = get_slice(img[sl, :, :, :], slpos, axis)
                im = ax.imshow(data, cmap=cmap, interpolation='gaussian', vmin=clim[0], vmax=clim[1])
                ax.axis('off')
    fig.tight_layout(pad=0)
    if cbar == True:
        add_colorbar(fig, im, all_ax, title, clim)
    else:
        if title is not None:
            fig.suptitle(title, color='white')
    plt.close()
    return fig


def sense(file, dset='sense', title=None, nrows=1, axis='z', slpos=0.5):
    """Plot a slice through each channel of a SENSE dataset

    Args:
        file (str): .h5 file to load
        dset (str): Dataset within the .h5 file (default 'sense')
        title (str): Plot title. Defaults to ''.
    """

    with h5py.File(file, 'r') as f:
        I = f[dset][:]

    [nz, ny, nx, nc] = np.shape(I)
    slpos = get_slpos(axis, nx, ny, nz, slpos)

    ncols = int(np.ceil(nc / nrows))

    fig, ax = plt.subplots(nrows, ncols, figsize=(
        3*ncols, 3*nrows), facecolor='black')

    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    smap = cm.ScalarMappable(norm=norm, cmap='cmr.infinity')

    pha = np.angle(I)
    mag = np.abs(I)
    lims = np.nanpercentile(mag, (2, 98))
    mag = np.clip((mag - lims[0]) / (lims[1] - lims[0]), 0, 1)

    for ic in range(nc):
        pha_slice = get_slice(pha[:, :, :, ic], slpos, axis)
        mag_slice = get_slice(mag[:, :, :, ic], slpos, axis)
        colorized = smap.to_rgba(pha_slice, alpha=1., bytes=False)[:, :, 0:3]
        colorized = colorized * mag_slice[:, :, None]
        if nrows == 1:
            if nc > 1:
                this_ax = ax[ic]
            else:
                this_ax = ax
        else:
            this_ax = ax[int(np.floor(ic / ncols)), ic % ncols]
        this_ax.imshow(colorized)
        this_ax.axis('off')
    fig.suptitle(title, color='white')
    fig.tight_layout(pad=0)
    plt.close()
    return fig


def diff(fnames, titles=None, dsets=['image'],
         axis='z', slpos=0.5, iv=0, ifr=0, ic=0,
         comp='mag', clim=None, cmap='gray',
         difflim=None, diffmap='cmr.iceburn',
         figsize=4):
    """Plot the difference between images

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
        imgs = [get_img(f, ii, iv, ic, comp, ds) for f, ds, ii in zip(files, dsets, ifr)]

        nI = len(imgs)

        if titles is not None:
            if len(titles) != len(fnames):
                warnings.warn('Number of titles and files did not match')
                return

        for ii in range(nI - 1):
            if (np.shape(imgs[ii]) != np.shape(imgs[ii+1])):
                warnings.warn('Image dimensions did not match')
                return

        [nz, ny, nx] = np.shape(imgs[0])
        slpos = get_slpos(axis, nx, ny, nz, slpos)

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

            difflim[1] = np.amax([np.abs(difflim[0]), np.abs(difflim[1])])
            difflim[0] = -difflim[1]

        fig, ax = plt.subplots(nI, nI, figsize=(
            nI*figsize, nI*figsize), facecolor='black')
        for ii in range(nI):
            imi = ax[ii, ii].imshow(slices[ii], cmap=cmap, interpolation='gaussian',
                                    vmin=clim[0], vmax=clim[1])
            ax[ii, ii].axis('off')
            if titles is not None:
                ax[ii, ii].text(0.5, 0.9, titles[ii], color='white',
                                transform=ax[ii, ii].transAxes, ha='center')
            for jj in range(ii):
                imd = ax[jj, ii].imshow(diffs[ii][jj], cmap=diffmap, interpolation='gaussian',
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


def diffL(fnames, titles=None, dsets=['image'],
         axis='z', slpos=0.5, iv=0, ifr=0, ic=0,
         comp='mag', clim=None, cmap='gray',
         difflim=None, diffmap='cmr.iceburn',
         figsize=8, interp='none', mode='progressive'):
    """Plot the difference between images

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
        imgs = [get_img(f, ii, iv, ic, comp, ds) for f, ds, ii in zip(files, dsets, ifr)]

        nI = len(imgs)

        if titles is not None:
            if len(titles) != len(fnames):
                warnings.warn('Number of titles and files did not match')
                return

        for ii in range(nI - 1):
            if (np.shape(imgs[ii]) != np.shape(imgs[ii+1])):
                warnings.warn('Image dimensions did not match')
                return

        [nz, ny, nx] = np.shape(imgs[0])
        slpos = get_slpos(axis, nx, ny, nz, slpos)

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
        for ii in range(nI - 1):
            if mode == 'progressive':
                index = ii
            else:
                index = 0
            diffs.append((slices[ii + 1] - slices[index]) * 100 / clim[1])

        if not difflim:
            difflim = (np.inf, -np.inf)
            for ii in range(nI - 1):
                templim = np.nanpercentile(diffs[ii], (2, 98))
                difflim = [np.amin([difflim[0], templim[0]]), np.amax([difflim[1], templim[1]])]

            difflim[1] = np.amax([np.abs(difflim[0]), np.abs(difflim[1])])
            difflim[0] = -difflim[1]

        fig, ax = plt.subplots(2, nI, figsize=(
            nI*figsize, 2*figsize), facecolor='black')
        for ii in range(nI):
            imi = ax[0, ii].imshow(slices[ii], cmap=cmap, interpolation=interp,
                                    vmin=clim[0], vmax=clim[1])
            ax[0, ii].axis('off')
            if titles is not None:
                ax[0, ii].text(0.1, 0.9, titles[ii], color='white',
                                transform=ax[0, ii].transAxes, ha='left')
            if ii > 0:
                imd = ax[1, ii].imshow(diffs[ii - 1], cmap=diffmap, interpolation=interp,
                                       vmin=difflim[0], vmax=difflim[1])
                ax[1, ii].axis('off')
            else:
                ax[1, ii].set_facecolor('black')
                ax[1, ii].axis('off')
        # axi = fig.add_axes([0.02, 0.2, 0.02, 0.6])
        fig.subplots_adjust(wspace=0, hspace=0)
        cbi = fig.colorbar(imi, ax=ax[0, :], location='left',
                           aspect=50, pad=0.01, shrink=0.8)
        cbi.ax.xaxis.set_tick_params(color='w', labelcolor='w')
        cbi.ax.yaxis.set_tick_params(color='w', labelcolor='w')
        # axd = fig.add_axes([0.98, 0.2, 0.02, 0.6])
        cbd = fig.colorbar(imd, ax=ax[1, :], location='left',
                           aspect=50, pad=0.01, shrink=0.8)
        cbd.ax.xaxis.set_tick_params(color='w', labelcolor='w')
        cbd.ax.yaxis.set_tick_params(color='w', labelcolor='w')

        plt.close()
        return fig
