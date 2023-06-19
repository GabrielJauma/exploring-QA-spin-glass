import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
# from palettable.cartocolors.sequential import BluGrn_7


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    ax = plt.gca() if ax is None else ax
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    lc.set_array(np.asarray(c))
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def export(fig_title):
    plt.gcf()
    plt.margins(0, 0)
    plt.savefig('Figures/' + fig_title, bbox_inches='tight', pad_inches=0)


def transparent_cmap(cmap_name, power):
    'The output is a new colormap called cmap_name_t'
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(cmap_name)(range(ncolors))

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1, ncolors) ** power

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=cmap_name + '_t', colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)


def colorbar(mappable, ticks):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
    plt.sca(last_axes)
    return cbar
