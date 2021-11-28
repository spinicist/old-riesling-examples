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


def single(file, dset='image', title=None, ix=None, iy=None, iz=None, iv=0, cmap='gray', vmin=None, vmax=None, component='mag'):
    """3 plane plot of 3D image data in an h5 file

    Args:
        file (str): Path to .h5 file
        dset (str): Dataset within h5 file to plot. Default "image"
        title (str, optional): Plot title. Defaults to ''.
        ix   (int, optional): Slice index for x-axis.
        iy   (int, optional): Slice index for y-axis.
        iz   (int, optional): Slice index for z-axis.
        iv   (int, optional): Volume to slice, default 0.
        cmap (str, optional): colormap. Defaults to 'gray'.
        vmin (float, optional): Lower window limit. Defaults to None.
        vmax (float, optional): Upper window limit. Defaults to None.
        component (str, opt): mag/pha/real/imaginary. Default mag
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]

    [nv, nz, ny, nx] = np.shape(I)
    if not (ix or iy or iz):
        ix = int(nx/2)
        iy = int(ny/2)
        iz = int(nz/2)

    fn = get_comp(component)
    img = fn(np.squeeze(I[iv, :, :, :]))
    lims = np.nanpercentile(img, (2, 98))
    if not vmin:
        vmin = lims[0]
    if not vmax:
        vmax = lims[1]

    fig, ax = plt.subplots(1, 3, figsize=(
        16, 6), facecolor='black', constrained_layout=True)
    ax[0].imshow(np.squeeze(img[iz, :, :]), cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].axis('image')
    ax[1].imshow(np.squeeze(img[:, iy, :]), cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].axis('image')
    im = ax[2].imshow(np.squeeze(img[:, :, ix]),
                      cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].axis('image')
    cb = fig.colorbar(im, ax=ax, location='right')
    cb.ax.xaxis.set_tick_params(color='w', labelcolor='w')
    cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.suptitle(title, color='white')
    plt.close()
    return fig


def multi(file, dset='basis-images', title=None, ix=None, iy=None, iz=None, iv=0, vmin=None, vmax=None):
    """3 plane plot of 3D basis image data in an h5 file

    Args:
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]
    if I.ndim == 4:
        I = I[np.newaxis, :, :, :, :]
    [nv, nz, ny, nx, npar] = np.shape(I)

    if not (ix or iy or iz):
        ix = int(nx/2)
        iy = int(ny/2)
        iz = int(nz/2)

    imgs = np.real(np.squeeze(I[iv, :, :, :, :]))
    imglist = [imgs[:, :, :, ii] for ii in range(npar)]
    lims = [np.nanpercentile(img, (2, 98)) for img in imglist]
    if not vmax:
        vmax = [lims[ii][1] for ii in range(npar)]
    if not vmin:
        vmin = [lims[ii][0] for ii in range(npar)]
        vmin = -np.maximum(np.absolute(vmin), vmax)
        vmax = np.maximum(np.absolute(vmin), vmax)

    fig, ax = plt.subplots(npar, 3, figsize=(16, 6*npar), facecolor='black')
    for ip in range(npar):
        if vmin[ip] < 0:
            cmap = 'cet_bkr'
        else:
            cmap = 'gray'
        ax[ip, 0].imshow(np.squeeze(imgs[iz, :, :, ip]),
                         cmap=cmap, vmin=vmin[ip], vmax=vmax[ip])
        ax[ip, 0].axis('off')
        ax[ip, 1].imshow(np.squeeze(imgs[:, iy, :, ip]),
                         cmap=cmap, vmin=vmin[ip], vmax=vmax[ip])
        ax[ip, 1].axis('off')
        im = ax[ip, 2].imshow(np.squeeze(imgs[:, :, ix, ip]),
                              cmap=cmap, vmin=vmin[ip], vmax=vmax[ip])
        ax[ip, 2].axis('off')
        cb = fig.colorbar(im, location='right', ax=ax[ip, 2])
        cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
    fig.suptitle(title, color='white')
    fig.tight_layout(pad=0)
    plt.close()
    return fig


def slices(file, dset='image', title=None, nrows=1, iz=None, cmap='gray', vmin=None, vmax=None, component='mag'):
    """Plot a slice through each volume of a dataset

    Args:
        file (str): .h5 file to load
        dset (str): Dataset within the .h5 file (default 'image')
        title (str, optional): Plot title. Defaults to ''.
        nrows (int, optional): Number of rows to plot slices over
        iv   (int, optional): Volume to slice, default 0.
        cmap (str, optional): colormap. Defaults to 'gray'.
        vmin (float, optional): Lower window limit. Defaults to None.
        vmax (float, optional): Upper window limit. Defaults to None.
        component (str, opt): mag/pha/real/imaginary. Default mag
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]

    [nv, nz, ny, nx] = np.shape(I)
    if not iz:
        iz = int(nz/2)

    ncols = int(np.floor(nv / nrows))

    fn = get_comp(component)
    img = fn(np.squeeze(I[:, iz, :, :]))
    lims = np.nanpercentile(img, (2, 98))
    if not vmin:
        vmin = lims[0]
    if not vmax:
        vmax = lims[1]

    fig, ax = plt.subplots(nrows, ncols, figsize=(
        3*ncols, 3*nrows), facecolor='black')

    for iv in range(nv):
        data = np.squeeze(img[iv, :, :])
        if nrows == 1:
            this_ax = ax[iv]
        else:
            this_ax = ax[int(np.floor(iv / ncols)), iv % ncols]
        this_ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
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
        title (str, optional): Plot title. Defaults to ''.
    """

    f = h5py.File(file, 'r')
    I = f[dset][:]

    [nz, ny, nx, nc] = np.shape(I)
    if not iz:
        iz = int(nz/2)

    ncols = int(np.floor(nc / nrows))

    fig, ax = plt.subplots(nrows, ncols, figsize=(
        3*ncols, 3*nrows), facecolor='black')

    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    smap = cm.ScalarMappable(norm=norm, cmap=cc.m_colorwheel)

    data = np.squeeze(I[iz, :, :, :])
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
            this_ax = ax[ic]
        else:
            this_ax = ax[int(np.floor(ic / ncols)), ic % ncols]
        this_ax.imshow(colorized)
        this_ax.axis('off')
    fig.tight_layout(pad=0)
    fig.suptitle(title, color='white')
    plt.close()
    return fig


def diff(file1, file2, dset='image', title1='Image 1', title2='Image 2', sli=2, iz=None, iv=0, cmap='gray', vmin=None, vmax=None, diffscale=1, component='mag'):
    """Plot the difference between two images

    Args:
        file1 (str): Path to first .h5 file
        file2 (str): Path to second .h5 file
        dset (str): Dataset within h5 file to plot. Default "image"
        title (str, optional): Plot title. Defaults to ''.
        sli  (int, optional): Slice axis (0='x',1='y',2='z')
        iz   (int, optional): Slice index for z-axis. Default 2.
        iv   (int, optional): Volume to slice, default 0.
        cmap (str, optional): colormap. Defaults to 'gray'.
        vmin (float, optional): Lower window limit. Defaults to None.
        vmax (float, optional): Upper window limit. Defaults to None.
        diffscale(float, optional): Amount to inflate diffs by
        component (str, opt): mag/pha/real/imaginary. Default mag
    """

    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')
    I1 = f1[dset][:]
    I2 = f2[dset][:]

    if (np.shape(I1) != np.shape(I2)):
        warnings.warn('Image dimensions did not match')
        return

    [nv, nz, ny, nx] = np.shape(I1)
    if not iz:
        iz = int(nz/2)

    fn = get_comp(component)
    if sli == 0:
        img1 = fn(np.squeeze(I1[iv, iz, :, :]))
        img2 = fn(np.squeeze(I2[iv, iz, :, :]))
    elif sli == 1:
        img1 = fn(np.squeeze(I1[iv, :, iz, :]))
        img2 = fn(np.squeeze(I2[iv, :, iz, :]))
    else:
        img1 = fn(np.squeeze(I1[iv, :, :, iz]))
        img2 = fn(np.squeeze(I2[iv, :, :, iz]))

    lims = np.nanpercentile(img1, (2, 98))
    if not vmin:
        vmin = lims[0]
    if not vmax:
        vmax = lims[1]

    fig, ax = plt.subplots(1, 3, figsize=(16, 6), facecolor='black')
    ax[0].imshow(img1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].axis('off')
    ax[0].set_title(title1, color='white')
    ax[1].imshow(img2, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].axis('off')
    ax[1].set_title(title2, color='white')
    diff = diffscale * (img2 - img1)
    ax[2].imshow(diff, cmap=cc.m_bkr, vmin=-vmax, vmax=vmax)
    ax[2].axis('off')
    ax[2].set_title(f'Diff x{diffscale}', color='white')
    fig.tight_layout(pad=0)
    plt.close()
    return fig
