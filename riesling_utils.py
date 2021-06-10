import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import h5py


def imshow3(I, ncol=None, nrow=None, cmap='gray', vmin=None, vmax=None, order='col'):
    """Multi-pane plot of 3D data

    Inspired by the matlab function imshow3.
    https://github.com/mikgroup/espirit-matlab-examples/blob/master/imshow3.m

    Expands the 3D data along the last dimension. Data is shown on the current matplotlib axis.

        - ncol: Number of columns
        - nrow: Number of rows
        - cmap: colormap ('gray')

    Output:
        - I3: Same image as shown

    Args:
        I (array): 3D array with 2D images stacked along last dimension
        ncol (int, optional): Number of columns. Defaults to None.
        nrow (int, optional): Number of rows. Defaults to None.
        cmap (str, optional): Colormap. Defaults to 'gray'.
        vmin (innt, optional): Lower window limit. Defaults to None.
        vmax (int, optional): Upper window limit. Defaults to None.
        order (str, optional): Plot order 'col/row'. Defaults to 'col'.

    Returns:
        array: Image expanded along the third dimension
    """

    [nx, ny, n] = np.shape(I)
    if (not nrow) and (not ncol):
        nrow = int(np.floor(np.sqrt(n)))
        ncol = int(n/nrow)
    elif not ncol:
        ncol = int(np.ceil(n/nrow))
    elif not nrow:
        nrow = int(np.ceil(n/ncol))

    I3 = np.zeros((ny*nrow, nx*ncol))

    i = 0
    if order == 'col':
        for ix in range(ncol):
            for iy in range(nrow):
                try:
                    I3[iy*ny:(iy+1)*ny, ix*nx:(ix+1)*nx] = I[:, :, i]
                except:
                    warnings.warn('Warning: Empty slice. Setting to 0 instead')
                    continue

                i += 1

    else:
        for iy in range(nrow):
            for ix in range(ncol):
                try:
                    I3[iy*ny:(iy+1)*ny, ix*nx:(ix+1)*nx] = I[:, :, i]
                except:
                    warnings.warn('Warning: Empty slice. Setting to 0 instead')
                    continue

                i += 1

    plt.imshow(I3, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')

    return I3


def read_radial_h5(f, vol):
    """
    Read in riesling radial data

    Input:
        f: .h5 file
        vol: Volume to read
    """

    f = h5py.File(f, 'r')

    # Reshape trajectory
    traj = f['trajectory'][:]

    # Reshape data
    data = f['data/%04.d' % vol][:]
    ds = data.shape
    data_rs = np.transpose(np.reshape(np.reshape(
        data, (1, np.prod(ds))), ds[::-1]), (2, 1, 0))

    info = f['info'][:]

    f.close()

    return data_rs, traj_rs, info


def plot_3plane(I, ix=None, iy=None, iz=None, cmap='gray', vmin=None, vmax=None):
    """3 plane plot of 3D image data

    Args:
        I (array): 3D image array
        title (str, optional): Plot title. Defaults to ''.
        cmap (str, optional): colormap. Defaults to 'gray'.
        vmin (int, optional): Lower window limit. Defaults to None.
        vmax (int, optional): Upper window limit. Defaults to None.
    """

    if not (ix or iy or iz):
        [nx, ny, nz] = np.shape(I)
        ix = int(nx/2)
        iy = int(ny/2)
        iz = int(nz/2)

    I = imshow3(np.stack([I[ix, :, :], I[:, iy, :], I[:, :, iz]], axis=2),
                ncol=3, nrow=1, cmap=cmap, vmin=vmin, vmax=vmax)

    return I


def load_nii(f):
    data = nib.load(f).get_fdata(dtype=np.complex64)

    # To be safe we read in complex but discard imag if
    # data is real
    if not np.any(np.iscomplex(data)):
        data = np.real(data)

    return np.squeeze(data)
