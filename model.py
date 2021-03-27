# Model definition
from yacs.config import CfgNode as CN

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

# TODO add chunking
PAD_TOKEN = -100
PAD_CAST = 0 # Inside the model, to have embedding layer accommodate pads, we set pads to 0 and offset others

class SeqSeqRNN(pl.LightningModule):
    NULL_INPUT = 0 # not to be confused with alphabet of 0, which is embedded.
    GO_CUE = -1
    SIZE_PER_INPUT = 2 # Embed to 2D (these are very low-dimensional to begin with.)

    def __init__(
        self,
        config: CN,
    ):
        assert config.MODEL.TYPE is 'gru', 'non-gru rnns unsupported'
        super().__init__()
        alphabet_sizes = config.TASK.ALPHABET_SIZES

        self.input_size = len(alphabet_sizes) * SeqSeqRNN.SIZE_PER_INPUT
        self.hidden_size = config.MODEL.HIDDEN_SIZE
        self.rnn = nn.GRU(self.input_size, self.hidden_size, 1)
        self.embedders = nn.ModuleList(
            [nn.Embedding(size + 1, SeqSeqRNN.SIZE_PER_INPUT, padding_idx=PAD_CAST) for size in alphabet_sizes]
        )

        self.go_cue = nn.Parameter(torch.rand(self.input_size))

        # self.encoder = self.rnn
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.hidden_size, size) for size in alphabet_sizes]
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.hold_period = config.MODEL.HOLD
        self.weight_decay = config.TRAIN.WEIGHT_DECAY

    def forward(self, input_seq, input_lengths):
        # A forward pass of this model will produce the output logits
        # args:
        #   input_seq - T x B x D (padded)
        #   input_lengths - B
        # returns:
        #   GRU outputs - T x B x H
        offset_seq = input_seq + 1
        offset_seq[input_seq == PAD_TOKEN] = PAD_CAST
        input_seqs = torch.unbind(offset_seq, dim=-1)
        embedded_seqs = [embed(seq) for embed, seq in zip(self.embedders, input_seqs)] # list of T x B x H
        embedded_seq = torch.cat(embedded_seqs, dim=-1).permute(1, 0, 2) # T x B x H -> B x T x H

        # Encode
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_seq, input_lengths.cpu(), enforce_sorted=False
        )
        # * Note. There seems to be a GPU-only bug with batch_first=True and enforce_sorted=False (so we permuted the input).
        # * The bug should probably be filed at some point

        hidden = torch.zeros((1, embedded_seq.size(1), self.hidden_size), device=self.device) # TBD whether requires_grad is needed
        _, hidden = self.rnn(packed, hidden) # * Discarding encoding outputs. (Don't bother suppressing)

        # Hold period
        if self.hold_period > 0:
            fixate_input = torch.zeros(self.hold_period, hidden.size(1), self.input_size, device=self.device)
            _, hidden = self.rnn(fixate_input, hidden) # * Discarding fixation outputs.

        # Decode
        go_input = torch.zeros_like(embedded_seq)
        go_input[0] = self.go_cue
        outputs, _ = self.rnn(go_input, hidden)
        return outputs

    def predict(self, x):
        # arg: x - T x D
        input_seq, input_lengths = x
        input_seq = input_seq.unsqueeze(0)
        input_lengths = input_lengths.unsqueeze(0)
        rnn_outputs = self(input_seq, input_lengths)
        predictions = [clf(rnn_outputs) for clf in self.classifiers] # T x H -> list of T x C
        predicted_units = [torch.argmax(pred, dim=-1) for pred in predictions] # list of T
        predicted_units = torch.stack(predicted_units, dim=-1).int() # T x 1 x D
        return predicted_units[:input_lengths[0]]

    def training_step(self, batch, batch_idx):
        # input_seq - T x B x D
        input_seq, input_lengths = batch
        rnn_outputs = self(input_seq, input_lengths)
        predictions = [clf(rnn_outputs) for clf in self.classifiers] # T x B x H -> list of T x B x C
        losses = torch.stack([self.criterion(pred.permute(1, 2, 0), input_seq[..., i]) for i, pred in enumerate(predictions)], dim=-1)
        loss = losses.mean()
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, input_lengths = batch
        rnn_outputs = self(input_seq, input_lengths)
        predictions = [clf(rnn_outputs) for clf in self.classifiers] # T x B x H -> list of T x B x C
        losses = torch.stack([self.criterion(pred.permute(1, 2, 0), input_seq[..., i]) for i, pred in enumerate(predictions)], dim=-1)
        loss = losses.mean()
        self.log('val_loss', loss, prog_bar=True)

        predicted_units = [torch.argmax(pred, dim=-1) for pred in predictions] # list of T
        predicted_items = torch.stack(predicted_units, dim=-1).int().permute(1, 0, 2) # T x B x D
        correct = torch.masked_select(predicted_items == input_seq, input_seq != -100) # We may need more granularity to debug
        # Mask out padding
        accuracy = correct.float().mean()
        self.log('val_accuracy', accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        input_seq, input_lengths = batch
        rnn_outputs = self(input_seq, input_lengths)
        predictions = [clf(rnn_outputs) for clf in self.classifiers] # T x B x H -> list of T x B x C
        losses = torch.stack([self.criterion(pred.permute(1, 2, 0), input_seq[..., i]) for i, pred in enumerate(predictions)], dim=-1)
        loss = losses.mean()
        self.log('test_loss', loss, prog_bar=True)

        predicted_units = [torch.argmax(pred, dim=-1) for pred in predictions] # list of T
        predicted_items = torch.stack(predicted_units, dim=-1).int().permute(1, 0, 2) # T x B x D
        correct = torch.masked_select(predicted_items == input_seq, input_seq != -100) # We may need more granularity to debug
        # Mask out padding
        accuracy = correct.float().mean()
        self.log('test_accuracy', accuracy)
        return loss

    def configure_optimizers(self):
        # Reduce LR on plateau as a reasonable default
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50),
            'monitor': 'val_loss'
        }