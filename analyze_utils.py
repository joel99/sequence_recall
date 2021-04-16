import os.path as osp
import seaborn as sns
import numpy as np
import numpy.linalg as linalg
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.utils.extmath import svd_flip
from sklearn.decomposition import PCA

def prep_plt(ax=None):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    LARGE_SIZE = 15
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-muted')
    plt.figure(figsize=(6,4))
    if ax is None:
        ax = plt.gca()
    spine_alpha = 0.3
    ax.spines['right'].set_alpha(spine_alpha)
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    ax.spines['top'].set_alpha(spine_alpha)
    ax.grid(alpha=0.25)
    plt.tight_layout()

def plot_to_axis(fig, ax, traj, colors=None, palette=None, normalize=True, lines=True, scatter=False, colorbar=False):
    # palette is color scheme, colors will override
    if palette is None:
        palette = sns.color_palette("flare", as_cmap=True)
    values = []
    for i in range(traj.shape[0]-1):
        values.append(i/(traj.shape[0] if normalize else 500))
        # colors.append(palette(values[-1]))
        color = colors[i] if colors is not None else palette(values[-1])
        if lines:
            ax.plot(
                *traj[i:i+2].T,
                color=color
            )
        if scatter:
            ax.scatter(
                *traj[i:i+1].T,
                color=color
            )

    ax.axis('off')
    if colorbar: # Only supports 1 call.
        norm = mpl.colors.Normalize(vmin=0,vmax=len(values))
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        fig.colorbar(sm, cax=cax,
            # ticks=values[::4],
            ticks=np.linspace(0, len(values), 8),
            # ticks=np.arange(len(values))[::4],
            # boundaries=np.arange(len(values))[::4],
            orientation='horizontal'
        )

def svd(jac):
    U, S, Vt = linalg.svd(jac, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    return U, S, Vt

def load_device(try_load=0):
    # try_load -- use this if you have a GPU alloc-ed (in interactive mode)
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda", min(try_load, torch.cuda.device_count() - 1))
    else:
        device = torch.device("cpu")
    print(f'Loaded: {device}')
    return device

def part_ratio(fps, n=30, pca=None):
    if pca is None:
        pca = PCA(n_components=min(n, len(fps)))
        pca.fit(fps)
    sv = np.array(pca.singular_values_)
    return sum(sv) ** 2 / sum(sv ** 2)

def loc_part_ratio(points, sample_size=20, k=50, seed=0): # From Aitken 2020 A2
    np.random.seed(seed)
    # points: N x H
    sampled = np.random.choice(len(points), sample_size)
    sampled = points[sampled] # S x H
    # get k nearest neighbors for each point
    loc_ratios = []

    for point in sampled:
        diffs = points - point # N x H
        distances = diffs.pow(2).sum(dim=1) # N
        knn = torch.argsort(distances)[:k]
        knn_points = points[knn] # K x H
        loc_ratios.append(part_ratio(knn_points))
        # Now do pca and argsort here
    return loc_ratios, sum(loc_ratios) / len(loc_ratios)

# From https://github.com/audeering/audtorch/blob/master/audtorch/metrics/functional.py
def pearsonr(
        x,
        y,
        batch_first=True,
):
    """
    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`
    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`
    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:
        .. math::
            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2
        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.
    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors
    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])
    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
    if len(x.shape) == 0:
        print(x)
    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr

