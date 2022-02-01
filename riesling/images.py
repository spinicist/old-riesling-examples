import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import colorcet as cc
import warnings
plt.rcParams['font.size'] = 20


def get_comp(component):
    if component == 'mag':
        fn = np.abs
    elif component == 'log':
        def fn(x): return np.log1p(np.abs(x))
    elif component == 'pha':
        fn = np.angle
    elif component == 'real':
        fn = np.real
    elif component == 'imag':
        fn = np.imag
    else:
        fn = np.real
        warnings.warn('Unknown component, taking real')
    return fn


def single(file, dset='image', title=None, pos=None, iv=0, ie=0, comp='mag', cmap='gray', clim=None):
    """3 plane plot of 3D image data in an h5 file

    Args:
        file  (str): Path to .h5 file
        dset  (str): Dataset within h5 file to plot. Default "image"
        title (str): Plot title. Defaults to ''
        pos   (int * 3): Slice indices
        iv    (int): Volume index
        ie    (int): Echo/basis index
        comp  (str): mag/pha/real/imaginary. Default mag
        cmap  (str): colormap. Defaults to 'gray'
        clim  (float * 2): Window limits. Defaults to (2,98) percentiles
    """

    fn = get_comp(comp)

    with h5py.File(file, 'r') as f:
        dsetw = f[dset]
        if dsetw.ndim == 4:
            # Channels only image, e.g. phantom
            [nz, ny, nx, ne] = dsetw.shape
            img = fn(dsetw[:, :, :, ie])
        else:
            # Assume 5D
            [nv, nz, ny, nx, ne] = dsetw.shape
            img = fn(dsetw[iv, :, :, :, ie])

    if not (pos):
        pos = (int(nx/2), int(ny/2), int(nz/2))
    if not clim:
        clim = np.nanpercentile(img, (2, 98))

    fig, ax = plt.subplots(1, 3, figsize=(
        16, 6), facecolor='black', constrained_layout=True)
    ax[0].imshow(np.squeeze(img[pos[2], :, :]),
                 cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[0].axis('image')
    ax[1].imshow(np.squeeze(img[:, pos[1], :]),
                 cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[1].axis('image')
    im = ax[2].imshow(np.squeeze(img[:, :, pos[0]]),
                      cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[2].axis('image')
    cb = fig.colorbar(im, ax=ax, location='right')
    cb.ax.xaxis.set_tick_params(color='w', labelcolor='w')
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.suptitle(title, color='white')
    plt.close()
    return fig


def multi(file, dset='basis-images', title=None, pos=None, iv=0, comp='mag', cmap=None, clim=None):
    """3 plane plot of 3D basis image data in an h5 file

    Args:
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]

    [nv, nz, ny, nx, ne] = np.shape(I)
    if not (pos):
        pos = (int(nx/2), int(ny/2), int(nz/2))

    fn = get_comp(comp)
    img = fn(np.squeeze(I[iv, :, :, :, :]))
    if not clim:
        clim = np.nanpercentile(img, (2, 98))
        if clim[0] < 0:
            clim[0] = -np.maximum(np.absolute(clim[0]), clim[1])
            if not cmap:
                cmap = 'cet_bkr'
    if not cmap:
        cmap = 'gray'

    fig, ax = plt.subplots(ne, 3, figsize=(16, 6*ne), facecolor='black')
    for ie in range(ne):
        ax[ie, 0].imshow(np.squeeze(img[pos[2], :, :, ie]),
                         cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
        ax[ie, 0].axis('off')
        ax[ie, 1].imshow(np.squeeze(img[:, pos[1], :, ie]),
                         cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
        ax[ie, 1].axis('off')
        im = ax[ie, 2].imshow(np.squeeze(img[:, :, pos[0], ie]),
                              cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
        ax[ie, 2].axis('off')
        cb = fig.colorbar(im, location='right', ax=ax[ie, 2])
        cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.suptitle(title, color='white')
    fig.tight_layout(pad=0)
    plt.close()
    return fig


def slices(file, dset='image', title=None, nrows=1, iz=None, ie=0, comp='mag', cmap='gray', clim=None):
    """Plot a slice through each volume of a dataset

    Args:
        file (str): .h5 file to load
        dset (str): Dataset within the .h5 file (default 'image')
        title (str): Plot title. Defaults to ''
        nrows (int): Number of rows to plot slices over
        iv   (int): Volume to slice, default 0
        ie   (int): Echo index, default 0
        cmap (str): colormap. Defaults to 'gray'.
        vmin (float): Lower window limit. Defaults to None.
        vmax (float): Upper window limit. Defaults to None.
        comp (str, opt): mag/pha/real/imaginary. Default mag
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]

    [nv, nz, ny, nx, ne] = np.shape(I)
    if not iz:
        iz = int(nz/2)

    ncols = int(np.ceil(nv / nrows))

    fn = get_comp(comp)
    img = fn(np.squeeze(I[:, iz, :, :, ie]))
    if not clim:
        clim = np.nanpercentile(img, (2, 98))

    fig, ax = plt.subplots(nrows, ncols, figsize=(
        3*ncols, 3*nrows), facecolor='black')

    for iv in range(nv):
        data = np.squeeze(img[iv, :, :])
        if nrows == 1:
            this_ax = ax[iv]
        else:
            this_ax = ax[int(np.floor(iv / ncols)), iv % ncols]
        this_ax.imshow(data, cmap=cmap,
                       vmin=clim[0], vmax=clim[1], origin='lower')
        this_ax.axis('off')
    fig.tight_layout(pad=0)
    fig.suptitle(title, color='white')
    plt.close()
    return fig


def sense(file, dset='sense', title=None, nrows=1, iz=None):
    """Plot a slice through each channel of a SENSE dataset

    Args:
        file (str): .h5 file to load
        dset (str): Dataset within the .h5 file (default 'sense')
        title (str): Plot title. Defaults to ''.
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]

    [nz, ny, nx, nc] = np.shape(I)
    if not iz:
        iz = int(nz/2)

    ncols = int(np.ceil(nc / nrows))

    fig, ax = plt.subplots(nrows, ncols, figsize=(
        3*ncols, 3*nrows), facecolor='black')

    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    smap = cm.ScalarMappable(norm=norm, cmap=cc.m_colorwheel)

    data = I[iz, :, :, :]
    pha = np.angle(data)
    mag = np.abs(data)
    lims = np.nanpercentile(mag, (2, 98))
    mag = np.clip((mag - lims[0]) / (lims[1] - lims[0]), 0, 1)

    for ic in range(nc):
        pha_slice = np.squeeze(pha[:, :, ic])
        mag_slice = np.squeeze(mag[:, :, ic])
        colorized = smap.to_rgba(pha_slice, alpha=1., bytes=False)[:, :, 0:3]
        colorized = colorized * mag_slice[:, :, None]
        if nrows == 1:
            if nc > 1:
                this_ax = ax[ic]
            else:
                this_ax = ax
        else:
            this_ax = ax[int(np.floor(ic / ncols)), ic % ncols]
        this_ax.imshow(colorized, origin='lower')
        this_ax.axis('off')
    fig.suptitle(title, color='white')
    fig.tight_layout(pad=0)
    plt.close()
    return fig


def diff(file1, file2, dset='image', title1='Image 1', title2='Image 2',
         sli=2, iz=None, iv=0, ie=0, comp='mag',
         cmap='gray', clim=None,
         diffscale=1, difflim=None):
    """Plot the difference between two images

    Args:
        file1 (str): Path to first .h5 file
        file2 (str): Path to second .h5 file
        dset (str): Dataset within h5 file to plot. Default "image"
        title (str): Plot title. Defaults to ''.
        sli  (int): Slice axis (0='x',1='y',2='z')
        iz   (int): Slice index for z-axis. Default 2.
        iv   (int): Volume to slice, default 0.
        cmap (str): colormap. Defaults to 'gray'.
        vmin (float): Lower window limit. Defaults to None.
        vmax (float): Upper window limit. Defaults to None.
        diffscale(float): Amount to inflate diffs by
        comp (str, opt): mag/pha/real/imaginary. Default mag
    """

    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')
    I1 = f1[dset][:]
    I2 = f2[dset][:]

    if (np.shape(I1) != np.shape(I2)):
        warnings.warn('Image dimensions did not match')
        return

    [nv, nz, ny, nx, ne] = np.shape(I1)
    if not iz:
        iz = int(nz/2)

    fn = get_comp(comp)
    if sli == 0:
        img1 = fn(np.squeeze(I1[iv, iz, :, :, ie]))
        img2 = fn(np.squeeze(I2[iv, iz, :, :, ie]))
    elif sli == 1:
        img1 = fn(np.squeeze(I1[iv, :, iz, :, ie]))
        img2 = fn(np.squeeze(I2[iv, :, iz, :, ie]))
    else:
        img1 = fn(np.squeeze(I1[iv, :, :, iz, ie]))
        img2 = fn(np.squeeze(I2[iv, :, :, iz, ie]))

    if not clim:
        clim1 = np.nanpercentile(img1, (2, 98))
        clim2 = np.nanpercentile(img2, (2, 98))
        clim = (np.amin((clim1[0], clim2[0])), np.amax((clim1[1], clim2[1])))
    diff = diffscale * (img2 - img1)
    if not difflim:
        difflim = np.nanpercentile(np.abs(diff), (2, 98))
        difflim = (-difflim[1], difflim[1])

    fig, ax = plt.subplots(1, 3, figsize=(16, 6), facecolor='black')
    ax[0].imshow(img1, cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[0].axis('off')
    ax[0].set_title(title1, color='white')
    ax[1].imshow(img2, cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[1].axis('off')
    ax[1].set_title(title2, color='white')
    diffim = ax[2].imshow(diff, cmap=cc.m_bkr, vmin=difflim[0],
                          vmax=difflim[1], origin='lower')
    ax[2].axis('off')
    ax[2].set_title(f'Diff x{diffscale}', color='white')
    cb = fig.colorbar(diffim, ax=ax[2], location='right')
    cb.ax.xaxis.set_tick_params(color='w', labelcolor='w')
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.tight_layout(pad=0)
    plt.close()
    return fig


def grid(file, dset='cartesian', title=None, pos=None, ic=0, ie=0, comp='log', cmap='plasma', clim=None):
    """3 plane plot of 3D image data in an h5 file

    Args:
        file  (str): Path to .h5 file
        dset  (str): Dataset within h5 file to plot. Default "image"
        title (str): Plot title. Defaults to ''
        pos   (int * 3): Slice indices
        iv    (int): Volume index
        ie    (int): Echo/basis index
        comp  (str): mag/pha/real/imaginary. Default mag
        cmap  (str): colormap. Defaults to 'gray'
        clim  (float * 2): Window limits. Defaults to (2,98) percentiles
    """

    with h5py.File(file, 'r') as f:
        I = f[dset][:]

    [nz, ny, nx, ne, nc] = np.shape(I)
    if not (pos):
        pos = (int(nx/2), int(ny/2), int(nz/2))

    fn = get_comp(comp)
    img = fn(np.squeeze(I[:, :, :, ie, ic]))
    if not clim:
        # There are a lot of zeros in typical cartesian grids. Use an expanded range
        clim = np.nanpercentile(img, (0, 100))
    fig, ax = plt.subplots(1, 3, figsize=(
        16, 6), facecolor='black', constrained_layout=True)
    ax[0].imshow(np.squeeze(img[pos[2], :, :]),
                 cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[0].axis('image')
    ax[1].imshow(np.squeeze(img[:, pos[1], :]),
                 cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[1].axis('image')
    im = ax[2].imshow(np.squeeze(img[:, :, pos[0]]),
                      cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[2].axis('image')
    cb = fig.colorbar(im, ax=ax, location='right')
    cb.ax.xaxis.set_tick_params(color='w', labelcolor='w')
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.suptitle(title, color='white')
    plt.close()
    return fig
