#%%
import numpy as np
import torch
from torch.utils.data import random_split

from dataset import get_hp_tag
# Generate sequences of various lengths, each item is a tuple of IDs.
# This process is simple to do on the fly; but we do this just to be safe about reproducibility.

# len(ALPHABET_SIZES) = D, dimensionality of each item to remember
ALPHABET_SIZES = [2]
# ALPHABET_SIZES = [4]
# ALPHABET_SIZES = [8]
# ALPHABET_SIZES = [12]
# ALPHABET_SIZES = [16]
# ALPHABET_SIZES = [20]
# ALPHABET_SIZES = [24]
# ALPHABET_SIZES = [28]
# ALPHABET_SIZES = [32]
# ALPHABET_SIZES = [10]

# 2c
# ALPHABET_SIZES = [64]
# ALPHABET_SIZES = [2, 32]
# ALPHABET_SIZES = [4, 16]
# ALPHABET_SIZES = [8, 8]
# ALPHABET_SIZES = [4, 4, 4]
# ALPHABET_SIZES = [2, 2, 2, 2, 2, 2]

START = 5
END = 8
OFFSET = 0
# OFFSET = 4
# OFFSET = 8
# OFFSET = 12
T_RANGE = np.arange(START, END + 1) + OFFSET
# NUM_TRIALS_PER_T = 1000
NUM_TRIALS_PER_T = 4000
# NUM_TRIALS_PER_T = 12000
NUM_TRIALS_PER_T = 24000
# NUM_TRIALS_PER_T = 48000

hp_tag = get_hp_tag(ALPHABET_SIZES, T_RANGE)
PAD_VALUE = -100

def generate_sequences(a_sizes, n, t, seed=0):
    # Generate up to n unique sequences
    torch.manual_seed(seed)
    seqs = [torch.randint(0, size, (n, t)) for size in a_sizes]
    seqs = torch.stack(seqs, dim=-1) # N x T x D
    return torch.unique(seqs, dim=0)

all_seqs = []
lengths = []
# Since dynamic range of sequence lengths is not enormous, pad here, instead of in iterator.
for t in T_RANGE:
    new_seqs = generate_sequences(ALPHABET_SIZES, NUM_TRIALS_PER_T, t)
    all_seqs.extend(new_seqs) # n' x t x d
    lengths.append(torch.full(new_seqs.size()[:1], t))
lengths = torch.cat(lengths)
all_seqs = torch.nn.utils.rnn.pad_sequence(all_seqs, padding_value=PAD_VALUE, batch_first=True)

print(all_seqs.size()) # N x T x D
print(lengths.size()) # N

indices = torch.randperm(len(lengths))
indices_train = indices[:int(len(indices) * 0.8)]
indices_test = indices[int(len(indices) * 0.8):]

lengths_train = lengths[indices_train]
all_seqs_train = all_seqs[indices_train]

torch.save({
    'sequences': all_seqs[indices_train],
    'lengths': lengths[indices_train]
}, f'data/{hp_tag}-train.pth')

torch.save({
    'sequences': all_seqs[indices_test],
    'lengths': lengths[indices_test]
}, f'data/{hp_tag}-test.pth')
