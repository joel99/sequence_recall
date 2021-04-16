#%%
import math
import numpy as np
import numpy.linalg as linalg

import torch
import pandas as pd
import os
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

import seaborn as sns
import PIL.Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config.default import get_config
from model import SeqSeqRNN
from dataset import SequenceRecallDataset

from analyze_utils import (
    prep_plt, plot_to_axis,
    load_device,
    loc_part_ratio, part_ratio, svd
)

from fp_finder import (
    load_recurrent_model,
    get_model_inputs, noise_ics, run_fp_finder, get_cache_path,
    get_jac_pt, get_jac_pt_sequential,
    get_eigvals, time_const
)

from fp_plotter import (
    get_pca_view,
    scatter_pca_points_with_labels,
    add_pca_inset,
    plot_spectrum
)

config = './config/e2a_alph2.yaml'
config = './config/e2a_alph8.yaml'
config = './config/e2a_alph32.yaml'
tag = ''
seed = 0
cn = get_config(config)
override_cache = True
override_cache = False

model = load_recurrent_model(config, seed)
cn = get_config(config)
dataset = SequenceRecallDataset(cn, split='test')

obs, ics, *_ = get_model_inputs(dataset, model, 2000) # get trajectories
mean_obs = obs

cache_fps = get_cache_path(config, seed, tag=tag)
if not osp.exists(cache_fps):
    run_fp_finder(config, seed, tag=tag, override=override_cache)
    cache_fps = get_cache_path(config, seed, tag=tag)

# TODO plot this with different color codes
# TODO take a look at other works

info = torch.load(cache_fps, map_location='cpu')
fps = info['fps']
deltas = info['deltas']
trajs = info['trajs']
input_seqs = info['input_seqs']
input_lengths = info['input_lengths']

pca = PCA(n_components=10)
all_trajs = torch.cat(trajs, dim=0)
pca.fit(all_trajs)

# pca.fit(fps)

labels = deltas > 9e-7
plt.scatter(deltas[~labels], np.random.rand(*deltas[~labels].shape), s=3)
plt.scatter(deltas[labels], np.random.rand(*deltas[labels].shape), s=3)
plt.xlim(0, 1e-5)
print(labels.size())

#%%

def angle_bn(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product) / (2 * math.pi) * 360


from scipy.linalg import subspace_angles

pca = PCA(n_components=10)
all_trajs = torch.cat(trajs, dim=0)
pca.fit(all_trajs)

pca2 = PCA(n_components=10)
pca2.fit(fps)

overlap_snippet = [0, 1, 2] # Not a great fit.
test_overlap = np.rad2deg(subspace_angles(pca.components_[overlap_snippet].T, pca2.components_[overlap_snippet].T))
print(test_overlap)
# test_overlap = np.rad2deg(subspace_angles(pca.components_[overlap_snippet], pca.components_[overlap_snippet]))

plt.plot(test_overlap, label='test')
plt.legend(frameon=False)

#%%
def unpad_flat_list(seqs, padded_length, lengths):
    # return list if lists of original seq
    # seqs - list to unpad
    # padded_length - full length
    # lengths - length of individual seqs
    pad_lengths = padded_length - lengths
    split_lengths = torch.stack([lengths, pad_lengths], dim=1).flatten().tolist()
    seq_chops = seqs.split(split_lengths)
    return seq_chops[::2]

def plot_trajs(
    f, trajs,
    traj_inds,
    input_seqs, input_lengths, color_scheme='time',
    view_indices=[0, 1], pca=None, ax=None, **kwargs
):
    if ax is None:
        ax = f.gca()
    if pca is None:
        all_trajs = torch.cat(trajs, dim=0)
        pca = PCA(n_components=10)
        pca.fit(all_trajs)

    sub_trajs = [trajs[i] for i in traj_inds]
    reduced_trajs = [pca.transform(s) for s in sub_trajs]
    # TODO more control in palette
    if color_scheme == 'time':
        colors = [None] * len(reduced_trajs)
    elif color_scheme == 'value': # I'm actually a clown for writing this
        seqs = [input_seqs[i] for i in traj_inds]
        lens = input_lengths[traj_inds]
        vals, inv = np.unique(torch.cat(seqs), return_inverse=True)
        palette_map = unpad_flat_list(torch.tensor(inv), input_seqs.size(1), lens)
        palette = sns.color_palette(n_colors=len(vals))
        colors = [[palette[i] for i in m] for m in palette_map]
        print(palette_map)
    for i, r in enumerate(reduced_trajs):
        plot_to_axis(
            f, ax, r[:, view_indices],
            lines=False, scatter=True,
            colors=colors[i], **kwargs
        )
    return ax

plot_trajs(plt.figure(), trajs,
    # [0],
    range(20),
    input_seqs, input_lengths, color_scheme = 'value',
    pca=pca
)

#%%
prep_plt()
plt.plot(np.cumsum(pca.explained_variance_ratio_))

#%%
def plot_fixed_points(
    f, points, pca, input_seqs, input_lengths, traj_inds=[], ax=None, view_indices=[0, 1], cbar=False, **kwargs
):
    ax, scatter, vals, colors = scatter_pca_points_with_labels(
        f,
        points,
        pca=pca,
        ax=ax,
        view_indices=view_indices,
        **kwargs
    )
    if cbar:
        f.colorbar(scatter, ax=ax)

    if colors is not None:
        patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, vals)]
        ax.legend(handles=patches, frameon=False, loc=(1.01, 0.2))
    plot_trajs(f, trajs, traj_inds, input_seqs, input_lengths, pca=pca, view_indices=view_indices)

    return ax


pca.fit(fps)
prep_plt()
plt.plot(np.cumsum(pca.explained_variance_ratio_))

#%%
# TODO uhh... plot based on trial timestep?
# Label based on output to produce (this has to be significant... right?)
# Wait, I can't label based on the final points, dude.

# All activity is far off fixed points. What if we project differently?
#   - I suppose -- if we actually went near fixed points, we'd be stuck.

# plot_fps = fps
plot_fps = fps[~labels]
# plot_deltas = deltas
plot_deltas = deltas[~labels]

def stability_map(point):
    # Returns number of eigenvalues with magnitude < 1 (quantifies topology)
    obs_grad, state_grad = get_jac_pt(model.rnn, mean_obs, point.unsqueeze(0))
    vals, vecs = get_eigvals(state_grad)
    # approx_stable = np.abs(vals) < (1 + 1e-4)
    approx_stable = np.abs(vals) < (1 - 1e-4)
    # approx_stable = (np.abs(vals) > (1 - 1e-4)) & (np.abs(vals) < (1 + 1e-4))
    return approx_stable.sum()

# By topology (approx)
c_map = [stability_map(pt) for pt in plot_fps]
# By speed
# c_map = torch.log(plot_deltas)

plot_fixed_points(
    plt.figure(),
    plot_fps,
    pca,
    input_seqs,
    input_lengths,
    # traj_inds=[],
    traj_inds=range(1),
    # traj_inds=range(10),
    view_indices=[0, 1],
    # labels=labels
    c=c_map,
    cbar=True
)

# plt.colorbar()
#%%
plt.hist(c_map)
# Vast majority of points have:
# - 2-5 dimensions which are > 1 (repel),
# - no neutral (memory) points
# - most are decay
# There are so many wih no stable modes? Is that even consistent with what I'm finding below?
# Oh, this means that every deflection will decay rapidly (or bounce off)

#%%
sample_count = 50
np.random.seed(0)

# Threshold test - FAST
is_fast = True
is_fast = False
subset = fps[labels if is_fast else ~labels]
selected_index = np.random.choice(subset.size(0), sample_count, replace=False)
selected_fps = subset[selected_index]

indices = [0, 1] # 2d
# * FYI you just ran something with the thresholded, ideally we see even more stable modes.
obs_grad, state_grad = get_jac_pt(model.rnn, mean_obs, selected_fps)

#%%
def plot_spectra(ax, state=True):
    prep_plt(ax)
    items = [plot_spectrum(j, ax) for j in (state_grad if state else obs_grad)]
    lines, spectra = zip(*items)

    stable = np.array([(np.abs(s - 1) < 0.01).sum() for s in spectra])
    ax.set_title(f"{tag} {'fast' if is_fast else 'slow'} FP J_{'rec' if state else 'in'} $\sigma$ n={len(lines)}")
    ax.text(
        0.2, 0.7,
        f'Stable span: {stable.mean():.3g} $\pm$ {stable.std():.3g}',
        transform=ax.transAxes
    )
    return lines
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
lines = plot_spectra(ax2, state=True)
# lines = plot_spectra(ax2, state=False)

# plot_fixed_points(fig, fps, pca, ax=ax)
# plot_fixed_points(fig, selected_fps, pca,
#     ax=ax, colors=[l.get_color() for l in lines], marker='^')


#%%
vals, vecs = get_eigvals(state_grad)
def plot_vals(spectrum, **kwargs):
    x = [x.real for x in spectrum]
    y = [x.imag for x in spectrum]
    sns.scatterplot(x, y, **kwargs)

print((time_const(vals) > 10).sum(axis=-1).mean())

#%%
time_consts = time_const(vals) # B
# print()
prep_plt()
for i in range(50):
# for i in range(50):
    if (time_const(vals[i]) > 1).all():
        # Wot, none of them are all out. WHat's happening?
        print('all out')
    plt.plot(time_const(vals[i]))
    # time = time_const(vals[i])
    # top3 = np.sort(time)[-3:]
    # plt.scatter([i] * 3, top3)
plt.yscale('log')
plt.ylabel('$\\tau$')
plt.xlabel('$\lambda$ (index)')
# plt.xlabel('FP # (index)')
plt.title('top 3 tau')
sns.despine()
plt.savefig('test.pdf', bbox_inches='tight')

#%%

#%%
# TODO why is there that spike? (almost every mode has time constant > 100 -- all dimensions are relevant?)
#%%
prep_plt()
for i in range(50):
# for i in range(10):
    plot_vals(vals[i], s=8, alpha=0.5)

plt.yticks(np.linspace(-0.4, 0.4, 5))
plt.xlabel('$\mathrm{Re}(\lambda)$')
plt.ylabel('$\mathrm{Im}(\lambda)$')
sns.despine()

plt.savefig('test.pdf', bbox_inches='tight')