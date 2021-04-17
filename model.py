# Model definition
from yacs.config import CfgNode as CN

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# TODO add chunking
PAD_TOKEN = -100

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
        self.alphabet_sizes = alphabet_sizes = config.TASK.ALPHABET_SIZES

        self.input_size = len(alphabet_sizes) * SeqSeqRNN.SIZE_PER_INPUT
        self.hidden_size = config.MODEL.HIDDEN_SIZE
        self.rnn = nn.GRU(self.input_size, self.hidden_size, 1)

        self.rehearse = config.MODEL.ALLOW_REHEARSAL
        self.chunk = config.MODEL.ALLOW_CHUNK
        self.pad_predict = config.MODEL.ALLOW_PAD_PREDICTION

        if self.rehearse:
            assert len(self.alphabet_sizes) == 1, 'not supported'
            self.PAD_CAST = alphabet_sizes[0] # No offset because it's annoying
        else:
            self.PAD_CAST = 0 # Inside the model, to have embedding layer accommodate pads, we set pads to 0 and offset others
            # We use zero so we have a constant token across alphabets
        self.embedders = nn.ModuleList(
            [nn.Embedding(size + 1, SeqSeqRNN.SIZE_PER_INPUT, padding_idx=self.PAD_CAST) for size in alphabet_sizes]
        )

        self.go_cue = nn.Parameter(torch.rand(self.input_size))

        pad_predict_option = 1 if self.pad_predict else 0
        if self.pad_predict:
            # We still don't use pad tokens in prediction loss, but allow its prediction as a "null element" for memory pruposes
            assert self.rehearse, "not supported"

        if self.chunk:
            assert not self.rehearse, "not supported"
            assert len(self.alphabet_sizes) == 1, "not supported"
            # e.g. allow length 2 chunks starting with item 0, i.e. [0, 1, 2] -> [0, 1, 2, (0,0), (0,1), (0,2)]
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.hidden_size, 2 * size + pad_predict_option) for size in alphabet_sizes]
            )
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(self.hidden_size, size + pad_predict_option) for size in alphabet_sizes]
            )
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, reduction='none')

        self.hold_period = config.MODEL.HOLD
        self.weight_decay = config.TRAIN.WEIGHT_DECAY

    def forward(self, input_seq, input_lengths):
        # A forward pass of this model will produce the output logits
        # args:
        #   input_seq - B x T x D (padded)
        #   input_lengths - B
        # returns:
        #   GRU outputs - T x B x H
        if self.rehearse:
            offset_seq = input_seq.clone()
        else:
            offset_seq = input_seq + 1
        offset_seq[input_seq == PAD_TOKEN] = self.PAD_CAST
        input_seqs = torch.unbind(offset_seq, dim=-1)
        embedded_seqs = [embed(seq) for embed, seq in zip(self.embedders, input_seqs)] # list of B x T x H
        embedded_seq = torch.cat(embedded_seqs, dim=-1).permute(1, 0, 2) # B x T x H -> T x B x H

        # Encode
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_seq, input_lengths.cpu(), enforce_sorted=False
        )
        # * Note. There seems to be a GPU-only bug with batch_first=True and enforce_sorted=False (so we permuted the input).
        # * The bug should probably be filed at some point

        hidden = torch.zeros((1, embedded_seq.size(1), self.hidden_size), device=self.device)
        outs, hidden = self.rnn(packed, hidden)

        # Hold period
        if self.hold_period > 0:
            if self.rehearse:
                def _re_embed(padded_states):
                    predictions = [clf(outs) for clf in self.classifiers] # T x B x H -> list of T x B x C
                    predictions = [torch.argmax(p, -1) for p in predictions] # list of T x B
                    predictions = torch.stack(predictions, dim=-1).long().permute(2, 0, 1) # D x T x B
                    re_embedded = [embed(seq) for embed, seq in zip(self.embedders, predictions)] # list of T x B x H
                    return torch.cat(re_embedded, -1) # i.e. send it back in

                if self.pad_predict:
                    unpacked, lens = pad_packed_sequence(outs)
                    gather_idx = (lens.to(self.device) - 1).unsqueeze(-1).expand(1, -1, unpacked.size(-1))
                    outs = torch.gather(unpacked, 0, gather_idx) # TBH -> 1BH
                    for step in range(self.hold_period):
                        reafferance = _re_embed(outs)
                        outs, hidden = self.rnn(reafferance, hidden)
                else: # a hackier version where we produce 'full rehearsals' at a time instead of stepwise
                    for step in range(self.hold_period // input_seq.size(1)):
                        outs, lens = pad_packed_sequence(outs)
                        reafferance = _re_embed(outs)
                        reafferance = pack_padded_sequence(reafferance, lens, enforce_sorted=False)
                        outs, hidden = self.rnn(reafferance, hidden)
            else:
                fixate_input = torch.zeros(self.hold_period, hidden.size(1), self.input_size, device=self.device)
                _, hidden = self.rnn(fixate_input, hidden) # * Discarding fixation outputs.

        # Decode
        go_input = torch.zeros_like(embedded_seq)
        go_input[0] = self.go_cue
        outputs, _ = self.rnn(go_input, hidden)
        return outputs

    def _get_chunked_candidates(self, seq):
        # Get chunked answers that serve as alternates for main seq answer
        # Currently, just chunks all the time, greedily (so at most 2 versions of the answer)
        # seq: T x 1 (raw label)
        # returns list[<T x 1]
        chunked_candidate = seq.new_full(seq.size(), PAD_TOKEN)
        seq_ptr = 0
        chunk_ptr = 0
        while seq_ptr < seq.size(0):
            if seq[seq_ptr] == 0 and not (seq_ptr + 1 == seq.size(0) or seq[seq_ptr + 1] == PAD_TOKEN):
                seq_ptr += 1
                chunked_candidate[chunk_ptr] = seq[seq_ptr] + self.alphabet_sizes[0]
            else:
                chunked_candidate[chunk_ptr] = seq[seq_ptr]
            seq_ptr += 1
            chunk_ptr += 1
        return chunked_candidate

    def _unchunk(self, seq):
        # T x 1 (multiple not supported) -> T' x 1
        # Split a prediction into an unrolled answer (for viz)
        unchunked = seq.new_full(seq.size(), PAD_TOKEN)
        seq_ptr = 0
        unchunked_ptr = 0
        while unchunked_ptr < seq.size(0):
            if seq[seq_ptr] <= self.alphabet_sizes[0]: # pad is 0, so 0th token is 1
                unchunked[unchunked_ptr] = seq[seq_ptr]
            else:
                unchunked[unchunked_ptr] = 1
                if unchunked_ptr < seq.size(0) - 1:
                    unchunked_ptr += 1
                    unchunked[unchunked_ptr] = seq[seq_ptr] - self.alphabet_sizes[0]
            seq_ptr += 1
            unchunked_ptr += 1
        return unchunked

    def predict(self, x):
        # arg: x - T x D
        input_seq, input_lengths = x
        input_seq = input_seq.unsqueeze(0)
        input_lengths = input_lengths.unsqueeze(0)
        rnn_outputs = self(input_seq, input_lengths)
        predictions = [clf(rnn_outputs) for clf in self.classifiers] # T x H -> list of T x C
        predicted_units = [torch.argmax(pred, dim=-1) for pred in predictions] # list of T
        predicted_items = torch.stack(predicted_units, dim=-1).int().permute(1, 0, 2) # B x T x D
        if self.chunk:
            # Whoops... gotta fix this.
            unchunked = [] # Unchunking may exceed length, clip to regular length
            for item in predicted_items:
                unchunked.append(self._unchunk(item))
            predicted_items = torch.stack(unchunked, 0)
        if self.pad_predict:
            predicted_items[predicted_items == self.alphabet_sizes[0]] = PAD_TOKEN

        return predicted_items[0][:input_lengths[0]]

    def _chunk_loss(self, input_seq, predictions, losses):
        if not self.chunk:
            return losses
        chunked_alts = torch.stack([self._get_chunked_candidates(seq) for seq in input_seq], 0) # B x T x c
        chunked_losses = torch.stack([self.criterion(pred.permute(1, 2, 0), chunked_alts[..., i]) for i, pred in enumerate(predictions)], dim=-1)
        losses = losses.flatten(1, -1).mean(-1) # Group minima along T x C, not B
        chunked_losses = chunked_losses.flatten(1, -1).mean(-1)
        return torch.minimum(losses, chunked_losses) # Do this instead of average to prevent blurring, intuitively...

    def training_step(self, batch, batch_idx):
        # input_seq - B x T x D
        input_seq, input_lengths = batch
        rnn_outputs = self(input_seq, input_lengths)
        predictions = [clf(rnn_outputs) for clf in self.classifiers] # B x T x H -> list of B x t x C
        losses = torch.stack([self.criterion(pred.permute(1, 2, 0), input_seq[..., i]) for i, pred in enumerate(predictions)], dim=-1)
        losses = self._chunk_loss(input_seq, predictions, losses)
        loss = losses.mean()
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, input_lengths = batch
        rnn_outputs = self(input_seq, input_lengths)
        predictions = [clf(rnn_outputs) for clf in self.classifiers] # B x T x H -> list of B x T x C
        losses = torch.stack([self.criterion(pred.permute(1, 2, 0), input_seq[..., i]) for i, pred in enumerate(predictions)], dim=-1)
        losses = self._chunk_loss(input_seq, predictions, losses)
        loss = losses.mean()
        self.log('val_loss', loss, prog_bar=True)

        predicted_units = [torch.argmax(pred, dim=-1) for pred in predictions] # list of T
        predicted_items = torch.stack(predicted_units, dim=-1).int().permute(1, 0, 2) # B x T x D
        if self.chunk:
            unchunked = []
            for item in predicted_items:
                unchunked.append(self._unchunk(item))
            predicted_items = torch.stack(unchunked, 0)
        correct = torch.masked_select(predicted_items == input_seq, input_seq != PAD_TOKEN) # We may need more granularity to debug
        # Mask out padding
        accuracy = correct.float().mean()
        self.log('val_accuracy', accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        input_seq, input_lengths = batch
        rnn_outputs = self(input_seq, input_lengths)
        predictions = [clf(rnn_outputs) for clf in self.classifiers] # B x T x H -> list of B x T x C
        losses = torch.stack([self.criterion(pred.permute(1, 2, 0), input_seq[..., i]) for i, pred in enumerate(predictions)], dim=-1)
        losses = self._chunk_loss(input_seq, predictions, losses)
        loss = losses.mean()
        self.log('test_loss', loss, prog_bar=True)

        predicted_units = [torch.argmax(pred, dim=-1) for pred in predictions] # list of T
        predicted_items = torch.stack(predicted_units, dim=-1).int().permute(1, 0, 2) # B x T x D
        if self.chunk:
            unchunked = []
            for item in predicted_items:
                unchunked.append(self._unchunk(item))
            predicted_items = torch.stack(unchunked, 0)
        correct = torch.masked_select(predicted_items == input_seq, input_seq != PAD_TOKEN) # We may need more granularity to debug
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