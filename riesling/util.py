import numpy as np
import warnings

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

def get_cmap(component):
    if component == 'mag':
        return 'gray'
    elif component == 'log':
        return 'cmr.ember'
    elif component == 'pha':
        return 'cmr.infinity'
    elif component == 'real':
        return 'cmr.iceburn'
    elif component == 'imag':
        return 'cmr.iceburn'
    else:
        warnings.warn('Unknown component, assume real')
        return 'cmr.iceburn'

def get_clim(img, component):
    if component == 'log':
        clim = np.nanpercentile(img, (2, 98))
    elif component == 'mag':
        clim = np.nanpercentile(img, (2, 98))
    else:
        clim = np.nanpercentile(img, (2, 98))
        if clim[0] < 0:
            clim[1] = np.amax([np.abs(clim[0]), np.abs(clim[1])])
            clim[0] = -clim[1]
    return clim

def get_slice(img, sl, axis):
    if axis == 'z':
        data = img[sl, :, :].T
    elif axis == 'y':
        data = img[:, sl, :].T
    else:
        data = np.flipud(img[:, :, sl])
    return data

def get_slpos(axis, nx, ny, nz, pos):
    if axis == 'z':
        slpos = int(np.floor(nz * pos))
    elif axis == 'y':
        slpos = int(np.floor(ny * pos))
    else:
        slpos = int(np.floor(nx * pos))
    return slpos

def add_colorbar(fig, im, ax, clabel, clims,
             black_backg=True, tick_fmt='{:.4g}', orient='v'):
    """
    Plots a colorbar in the specified axes

    Parameters:

    - axes -- matplotlib axes instance to use for plotting
    - cm_name -- Colormap name
    - clims -- The limits for the colormap & bar
    - clabel -- Label to place on the color axis
    - black_bg -- Boolean indicating if the background to this plot is black, and hence white text/borders should be used
    - show_ticks -- Set to false if you don't want ticks on the color axis
    - tick_fmt -- Valid format string for the tick labels
    - orient -- 'v' or 'h' for whether you want a vertical or horizontal colorbar
    """
    cb = fig.colorbar(im, location='right', ax=ax, aspect=50, pad=0.01, shrink=0.8)
    if black_backg:
        forecolor = 'w'
    else:
        forecolor = 'k'
    axes = cb.ax

    ticks = (clims[0], np.sum(clims)/2, clims[1])
    labels = (tick_fmt.format(clims[0]), clabel, tick_fmt.format(clims[1]))
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    if orient == 'h':
        rot=0
    else:
        rot=90
        cb.ax.get_yticklabels()[0].set_va('bottom')
        cb.ax.get_yticklabels()[1].set_va('center')
        cb.ax.get_yticklabels()[2].set_va('top')
    cb.ax.tick_params(labelrotation=rot, color='w', labelcolor='w')
