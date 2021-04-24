#%%
import os.path as osp
from yacs.config import CfgNode as CN
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import torch

from config.default import get_config
from model import SeqSeqRNN
from dataset import SequenceRecallDataset
from sklearn.metrics import confusion_matrix

# We need to grab the checkpoint, and its corresponding config. We can skip on the config
# TODO figure out how to store the config in checkpoint and just load checkpoints
config = './config/base.yaml'
# config = './config/large.yaml'
seed = 0
version = 1

def get_ckpt(config_path, seed, version):
    variant = osp.split(config_path)[1].split('.')[0]
    config = get_config(config_path)
    root = Path(f'runs/{variant}-{seed}/lightning_logs/')
    run_path = sorted(root.iterdir(), key=osp.getmtime)[-1]
    # * This is the default output, if you want to play around with a different checkpoint load it here.
    model_ckpt = list(run_path.joinpath('checkpoints').glob("*"))[0]
    return model_ckpt, config, variant
model_ckpt, config, _ = get_ckpt(config, seed, version)

def load_model(model_ckpt, config):
    weights = torch.load(model_ckpt, map_location='cpu')
    model = SeqSeqRNN(config)
    model.load_state_dict(weights['state_dict'])
    model.eval()
    return model

model = load_model(model_ckpt, config)
#%%
model_info = [
    ('./config/e5_no_chunk_l5-8.yaml', 0, 1),
    ('./config/e5_no_chunk_l9-12.yaml', 0, 1),
    ('./config/e5_no_chunk_l13-16.yaml', 0, 1),
    ('./config/e5_no_chunk_l17-20.yaml', 0, 1),
]

def get_error_pos(model, config):
    test_dataset = SequenceRecallDataset(config, split="test")
    all_errors = []
    for item, l in test_dataset:
        pred = model.predict((item, l)).squeeze()
        time_period = np.linspace(0, 1, l)
        error = time_period[pred != item[:l, 0]]
        all_errors.append(error)
    all_errors = np.concatenate(all_errors, 0)
    return all_errors

all_df = []
for path, seed, version in model_info:
    model_ckpt, config, variant = get_ckpt(path, seed, version)
    model = load_model(model_ckpt, config)
    errors = get_error_pos(model, config)
    annotate = variant.split('_')[-1][1:]
    all_df.append(
        pd.DataFrame({"variant": annotate, "error": pd.Series(errors)})
    )
all_df = pd.concat(all_df)

#%%
from analyze_utils import prep_plt
fig = plt.figure()
ax = fig.gca()
ax = sns.histplot(ax=ax, x="error", hue="variant", data=all_df, common_norm=False, stat='probability', multiple='dodge',
    bins=10,
    kde=True,
    alpha=0.3,
)
prep_plt(ax=ax)
sns.despine(ax=ax)
# sns.displot(x="error", hue="variant", kind="kde", data=all_df, cut=True, common_norm=False)

# sns.displot(x=all_errors, bins=10)
ax.legend(
    ["5-8", "9-12", "13-16", "17-20"],
    frameon=False,
    fontsize=14,
    title="Trial Length",
    title_fontsize=16,
)
ax.set_xlabel('Trial Position')

fig.savefig('test.pdf')

#%%
# Umm...
# Just plot the results.
exp_rehearse = pd.concat([
    pd.DataFrame({
        'Variant': 'No Rehearse',
        'Hold Period': pd.Series([16, 16, 32, 32, 64, 64, 128, 128,]),
        'seed': pd.Series([0, 1, 0, 1, 0, 1, 0, 1]),
        'Accuracy': pd.Series([
            0.698, 0.735, 0.645, 0.663, 0.38, 0.376, 0.2651, 0.27
        ])
    }),
    pd.DataFrame({
        'Variant': 'Partial Rehearse',
        'Hold Period': pd.Series([16, 16, 32, 32, 64, 64, 128, 128,]),
        'seed': pd.Series([0, 1, 0, 1, 0, 1, 0, 1]),
        'Accuracy': pd.Series([
            0.886, 0.724, 0.795, 0.834, 0.67, .63, 0.495, 0.472
        ])
    }),
    pd.DataFrame({
        'Variant': 'Full Rehearse',
        'Hold Period': pd.Series([16, 16, 32, 32, 64, 64, 128, 128,]),
        'seed': pd.Series([0, 1, 0, 1, 0, 1, 0, 1]),
        'Accuracy': pd.Series([
            0.897,0.946, 0.765, 0.774, 0.692, 0.845, 0.418, 0.675
        ])
    }),
])

fig = plt.figure()
ax = fig.gca()
ax = sns.lineplot(x='Hold Period', y='Accuracy', hue='Variant', data=exp_rehearse, ax=ax)

ax.legend(
    ["None", "Partial", "Full"],
    frameon=False,
    fontsize=14,
    title="Rehearsal",
    title_fontsize=16,
)
sns.despine(ax=ax)
fig.savefig('test.pdf')
