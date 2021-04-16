# Plotting Utils
import numpy as np
import numpy.linalg as linalg

import torch
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import seaborn as sns

from analyze_utils import (
    prep_plt, plot_to_axis,
    load_device,
    loc_part_ratio, part_ratio, svd
)

from fp_finder import (
    get_model_inputs, noise_ics, get_cache_path, get_jac_pt
)

def get_pca_view(points, pca=None, view_indices=[0, 1]):
    points = points.cpu()
    if pca is None:
        pca = PCA(n_components=30) # low because we're viz-ing
        pca.fit(points)
    return pca.transform(points)[:, view_indices], pca

def scatter_pca_points_with_labels(
    fig,
    points, # [N, H]
    pca=None,
    view_indices=[0, 1],
    ax=None,
    plotted=2000, # Slicing. This is a final step, all args should be in full.
    labels=None, # categorical labels for each point [N]
    palette=None,
    s=3,
    **kwargs, # Forward to scatter
):
    # Returns plotted axis, scatter patch, categories plotted and their colors.
    assert len(view_indices) in [2,3], "only 2d or 3d view supported"
    if ax is None:
        if len(view_indices) == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = plt.gca()
    reduced, pca = get_pca_view(points, pca=pca, view_indices=view_indices)
    reduced = reduced[:plotted]

    if labels is None:
        scatter = ax.scatter(*reduced.T, s=s, **kwargs) # Sparser plotting
        return ax, scatter, None, None
    else:
        unique, inverse = np.unique(labels[:plotted], return_inverse=True)
        if palette is None:
            palette = sns.color_palette('hls', n_colors=len(unique))
        colors = [palette[c] for c in inverse]
        scatter = ax.scatter(*reduced.T, color=colors, s=s, **kwargs)

    ax.axis('off')
    return ax, scatter, unique, palette

def add_pca_inset(
    fig,
    pca,
    loc=[0.25, 0.15, 0.2, 0.2],
    ax=None,
):
    if ax is None:
        ax = fig.add_axes(loc)
    ax.plot(np.arange(len(pca.components_)), np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel("# of PCs")
    ax.set_ylabel("VAF")
    ax.set_xticks([0, 3, 20])
    prep_plt(ax)
    sns.despine(ax=ax, right=False, top=False)
    return ax

def plot_spectrum(jac, ax=None):
    U, S, Vt = svd(jac)
    if ax is None:
        ax = plt.gca()
    return ax.plot(S, label="$\sigma$")[0], np.array(S)