import pathlib
import os.path as osp
from yacs.config import CfgNode as CN
import torch
from torch.utils.data import Dataset

def get_hp_tag(alphabet_sizes, t_range):
    alphabet_tag = "_".join(map(str, alphabet_sizes))
    return f"{alphabet_tag}-low_{t_range[0]}-high_{t_range[-1]}"

class SequenceRecallDataset(Dataset):
    def __init__(
        self,
        config,
        split="train",
        dataset_root="./data/",
    ):
        super().__init__()
        hp_tag = get_hp_tag(config.TASK.ALPHABET_SIZES, config.TASK.T_RANGE)
        data = torch.load(osp.join(dataset_root, f"{hp_tag}-{split}.pth"))
        self.sequences = data['sequences']
        self.lengths = data['lengths']

    def __len__(self):
        r""" Number of samples. """
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.lengths[index]
