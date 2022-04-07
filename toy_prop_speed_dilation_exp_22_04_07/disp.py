from copy import deepcopy
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def set_font_size(ax, font_size, legend_font_size=None):
    """Set font_size of all axis text objects to specified value."""

    texts = [ax.title, ax.xaxis.label, ax.yaxis.label] + \
        ax.get_xticklabels() + ax.get_yticklabels()

    for text in texts:
        text.set_fontsize(font_size)

    if ax.get_legend():
        if not legend_font_size:
            legend_font_size = font_size
        for text in ax.get_legend().get_texts():
            text.set_fontsize(legend_font_size)
            
            
def set_n_x_ticks(ax, n, x_min=None, x_max=None):
    x_ticks = ax.get_xticks()
    
    x_min = np.min(x_ticks) if x_min is None else x_min
    x_max = np.max(x_ticks) if x_max is None else x_max
    
    ax.set_xticks(np.linspace(x_min, x_max, n))
    
    
def set_n_y_ticks(ax, n, y_min=None, y_max=None):
    y_ticks = ax.get_yticks()
    
    y_min = np.min(y_ticks) if y_min is None else y_min
    y_max = np.max(y_ticks) if y_max is None else y_max
    
    ax.set_yticks(np.linspace(y_min, y_max, n))

    
def set_color(ax, color, box=False):
    """Set colors on all parts of axis."""

    if box:
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['left'].set_color(color)
    ax.spines['right'].set_color(color)

    ax.tick_params(axis='x', color=color)
    ax.tick_params(axis='y', color=color)

    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color(color)

    ax.title.set_color(color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    
    
def get_spaced_colors(cmap, n, step):
    """step from 0 to 1"""
    cmap = cm.get_cmap(cmap)
    return cmap((np.arange(n, dtype=float)*step)%1)


def get_ordered_colors(cmap, n, lb=0, ub=1):
    cmap = cm.get_cmap(cmap)
    return cmap(np.linspace(lb, ub, n))

    
def fast_fig(n_ax, ax_size, fig_w=15):
    """Quickly make figure and axes objects from number of axes and ax size (h, w)."""
    n_col = int(round(fig_w/ax_size[1]))
    n_row = int(np.ceil(n_ax/n_col))
    
    fig_h = n_row*ax_size[0]
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(fig_w, fig_h), tight_layout=True, squeeze=False)
    return fig, axs.flatten()


def graph_weight_matrix(mat, title, v_max=None, ax=None, cmap='hot', figsize=None):
    mat = copy(mat)

    if figsize is None:
        figsize = (4, 4)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    else:
        fig = None
    
    im = ax.matshow(mat, vmin=0, vmax=v_max if v_max is not None else mat.max(), cmap=cmap)
    plt.colorbar(im, cax=ax)

    ax.set_title(title)
    set_font_size(ax, 14)
    if fig:
        return ax, fig
    return ax


def graph_weights(w_r, w_u, v_max=None):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    
    graph_weight_matrix(w_r['E'], 'W_R[E]\n', v_max=v_max, ax=axs[0, 0])
    graph_weight_matrix(w_r['I'], 'W_R[I]\n', v_max=v_max, ax=axs[0, 1])
    graph_weight_matrix(w_u['E'], 'W_U[E]\n', v_max=v_max, ax=axs[1, 0])
    graph_weight_matrix(w_u['I'], 'W_U[I]\n', v_max=v_max, ax=axs[1, 1])
