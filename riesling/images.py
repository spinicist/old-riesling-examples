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


def planes(file, dset='image', title=None, pos=None, iv=0, ie=0, comp='mag', cmap='gray', clim=None):
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
        16, 6), facecolor='black')
    ax[0].imshow(np.squeeze(img[pos[2], :, :]).T,
                 cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax[0].axis('image')
    ax[1].imshow(np.squeeze(img[:, pos[1], :]).T,
                 cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax[1].axis('image')
    im = ax[2].imshow(np.squeeze(img[:, :, pos[0]]).T,
                      cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax[2].axis('image')
    cb = fig.colorbar(im, location='right', ax=ax[2])
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.suptitle(title, color='white')
    fig.tight_layout(pad=0)
    plt.close()
    return fig


def slices(file, dset='image', title=None, ie=0, iv=0,
           nslice=4, nrows=1, axis='z', start=None, stop=None,
           comp='mag', cmap='gray', clim=None):
    """Plot slices along one axis

    Args:
        file (str): .h5 file to load
        dset (str): Dataset within the .h5 file (default 'image')
        title (str): Plot title. Defaults to ''
        nslice (int): Number of slices
        nrows (int): Number of rows to plot slices over
        axis (str) : 'x' 'y' or 'z'
        ie   (int): Echo index, default 0
        iv   (int): Volume to slice, default 0
        comp (str, opt): mag/pha/real/imaginary. Default mag
        cmap (str): colormap. Defaults to 'gray'.
        clim (float, float): Lower/upper window limits
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]
    fn = get_comp(comp)
    img = fn(I[iv, :, :, :, ie])
    if not clim:
        clim = np.nanpercentile(img, (2, 98))
    [nz, ny, nx] = np.shape(img)

    if axis == 'z':
        maxn = nz
    elif axis == 'y':
        maxn = ny
    else:
        maxn = nx
    if not start:
        start = maxn / 4
    if not stop:
        end = 3 * maxn / 4
    slpos = np.floor(np.linspace(start, stop, nslice, endpoint=True))

    ncols = int(np.ceil(nslice / nrows))
    fig, all_ax = plt.subplots(nrows, ncols, figsize=(
        3*ncols, 3*nrows), facecolor='black')

    for ir in range(nrows):
        if nrows > 1:
            sl_ax = ax[ir, :]
        else:
            sl_ax = all_ax
        for ic in range(ncols):
            sl = (ir * ncols) + ic
            if sl < nslice:
                ax = sl_ax[ic]
                if axis == 'z':
                    data = np.squeeze(img[slpos[sl], :, :])
                elif axis == 'y':
                    data = np.squeeze(img[:, slpos[sl], :])
                else:
                    data = np.squeeze(img[:, :, slpos[sl]])
                im = ax.imshow(data, cmap=cmap,
                               vmin=clim[0], vmax=clim[1], origin='lower')
                ax.axis('off')
    cb = fig.colorbar(im, location='right')
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.tight_layout(pad=0)
    fig.suptitle(title, color='white')
    plt.close()
    return fig


def series(file, dset='image', title=None, which='echoes', io=0,
           nrows=1, axis='z', slpos=None,
           comp='mag', cmap='gray', clim=None):
    """One slice through echoes or volumes

    Args:
        file (str): .h5 file to load
        dset (str): Dataset within the .h5 file (default 'image')
        title (str): Plot title. Defaults to ''
        which (str): 'echoes' or 'vols'
        io (int): Index into 'other' dim (echoes/vols)
        nrows (int): Number of rows to plot slices over
        axis (str) : 'x' 'y' or 'z'
        slpos (int): Which slice
        comp (str, opt): mag/pha/real/imaginary. Default mag
        cmap (str): colormap. Defaults to 'gray'.
        clim (float, float): Lower/upper window limits
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]
    [nv, nz, ny, nx, ne] = np.shape(I)
    fn = get_comp(comp)
    if which == 'echoes':
        nslice = ne
        if axis == 'z':
            if not slpos:
                slpos = int(np.floor(nz / 2))
            img = fn(I[io, slpos, :, :, :]).transpose((2, 0, 1))
        if axis == 'y':
            if not slpos:
                slpos = int(np.floor(ny / 2))
            img = fn(I[io, :, slpos, :, :]).transpose((2, 0, 1))
        if axis == 'x':
            if not slpos:
                slpos = int(np.floor(nx / 2))
            img = fn(I[io, :, :, slpos, :]).transpose((2, 0, 1))
    else:
        nslice = nv
        if axis == 'z':
            if not slpos:
                slpos = nz / 2
            img = fn(I[:, slpos, :, :, io])
        if axis == 'y':
            if not slpos:
                slpos = ny / 2
            img = fn(I[:, :, slpos, :, io])
        if axis == 'x':
            if not slpos:
                slpos = nx / 2
            img = fn(I[:, :, :, slpos, io])

    if not clim:
        clim = np.nanpercentile(img, (2, 98))
        if clim[0] < 0:
            clim[1] = np.max(np.abs(clim))
            clim[0] = -clim[1]

    ncols = int(np.ceil(nslice / nrows))
    fig, all_ax = plt.subplots(nrows, ncols, figsize=(
        3*ncols, 3*nrows), facecolor='black')

    for ir in range(nrows):
        if nrows > 1:
            sl_ax = all_ax[ir, :]
        else:
            sl_ax = all_ax
        for ic in range(ncols):
            sl = (ir * ncols) + ic
            if sl < nslice:
                ax = sl_ax[ic]
                data = np.squeeze(img[sl, :, :])
                im = ax.imshow(data.T, cmap=cmap,
                               vmin=clim[0], vmax=clim[1])
                ax.axis('off')
    fig.tight_layout(pad=0)
    cb = fig.colorbar(im, location='right', ax=all_ax)
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    cb.ax.xaxis.set_tick_params(color='w', labelcolor='w')
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
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), facecolor='black')
    ax[0, 0].imshow(np.squeeze(img[pos[2], :, :]),
                    cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[0, 0].axis('image')
    ax[0, 1].imshow(np.squeeze(img[:, pos[1], :]),
                    cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[0, 1].axis('image')
    im = ax[1, 0].imshow(np.squeeze(img[:, :, pos[0]]),
                         cmap=cmap, vmin=clim[0], vmax=clim[1], origin='lower')
    ax[1, 0].axis('image')
    ax[1, 1].set_facecolor('k')
    cb = fig.colorbar(im, ax=ax[1, 1], location='right')
    cb.ax.xaxis.set_tick_params(color='w', labelcolor='w')
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.tight_layout(pad=0)
    fig.suptitle(title, color='white')
    plt.close()
    return fig
