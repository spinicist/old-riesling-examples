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

def get_slice(img, sl, axis):
    if axis == 'z':
        data = img[sl, :, :].T
    elif axis == 'y':
        data = img[:, sl, :].T
    else:
        data = img[:, :, sl].T
    return data

def mid_slice(axis, nx, ny, nz):
    if axis == 'z':
        slpos = int(np.floor(nz / 2))
    elif axis == 'y':
        slpos = int(np.floor(ny / 2))
    else:
        slpos = int(np.floor(nx / 2))
    return slpos

def sym_lim(img):
    lim = np.nanpercentile(img, (2, 98))
    if lim[0] < 0:
        lim[1] = np.amax([np.abs(lim[0]), np.abs(lim[1])])
        lim[0] = -lim[1]
    return lim

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
