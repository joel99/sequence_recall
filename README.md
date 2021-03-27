# Sequence Recall
How do RNNs maintain short-term memory in its sustained activity? How does performance compare to human performance?
We study this using a simple sequence recall task, analyzing how performance varies along several axes (while keeping human performance in mind).
After a quantitative study of performance, we also apply a fixed point analysis to learned dynamic structure.

Task:
- The model is fed a sequence of tokens (e.g. '1', '6', '3') of variable length, concluding with an `<EOS>` token.
- Immediately after the `<EOS>` token, the model must reproduce the input sequence.
- We may also study related settings e.g. fixed length seqeuences, delayed output.

Model:
- We use a standard 1-layer RNN-based sequence to sequence model. For simplicity, we will use the same RNN in the encoder and decoder and we will not feed outputs back into the RNN.

Code Layout:
- Codebase is written on `pytorch-lightning`.
- Datasets synthesized in `dataset_gen.py`, loaded in `dataset.py`
- Models in `model.py`
- Driver in `train.py`
- Analysis + evaluation sanity checks in `analysis.py`.

- `data` contains some basic datasets. Don't push here often.
- `shared_ckpts` contains checkpoints of good models to be used for fixed point analysis. Same as `data`, don't push here often.

Fixed point work will be loaded in a separate folder, TBD.

## Running
Train a model using `python train.py -c ./config/<variant_here>.yaml`