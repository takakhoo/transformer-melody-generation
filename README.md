# Transformer Melody Generation

A small, readable encoder–decoder Transformer that learns symbolic melodies and
generates continuations autoregressively. The project implements the model,
tokenization pipeline, masked training loss, and inference loop directly in
TensorFlow/Keras.

## Why this project

Musical notes can be modeled as a sequence in the same way as words or other
discrete tokens. This repository uses that analogy to make the core Transformer
building blocks tangible without hiding them behind a high-level model API.

## Architecture

- Sinusoidal positional encodings preserve note order.
- Multi-head self-attention models relationships across a melody.
- Encoder and decoder stacks use residual connections, normalization, dropout,
  and position-wise feed-forward layers.
- Padding-aware cross-entropy prevents padded positions from affecting loss.
- Greedy autoregressive decoding extends a user-provided seed phrase.

Melodies are represented as pitch-duration tokens such as `C4-1.0` (middle C,
one quarter-note beat).

## Quick start

```bash
git clone https://github.com/takakhoo/Transformer-Melody-Generation.git
cd Transformer-Melody-Generation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python train.py
```

`train.py` trains on `dataset.json`, then generates a melody from the seed
defined at the bottom of the file. Adjust `EPOCHS`, `BATCH_SIZE`, or the model
dimensions near the top of `train.py` for experiments.

The full default run is intentionally small enough for a CPU smoke test. On a
recent Apple Silicon laptop it completes ten epochs in seconds; exact loss and
generated notes vary because training is not seeded.

## Repository map

| File | Responsibility |
| --- | --- |
| `transformer.py` | Attention layers, positional encoding, encoder, decoder, and model |
| `melodypreprocessor.py` | Tokenization, sequence padding, and `tf.data` creation |
| `melodygenerator.py` | Greedy autoregressive generation and token decoding |
| `train.py` | Training loop, masked loss, configuration, and example inference |
| `dataset.json` | Small symbolic-melody training set |

## Current limitations

- The bundled dataset is intentionally small, so output quality is a learning
  demonstration rather than a production music model.
- Generation uses greedy decoding; temperature, top-k, and nucleus sampling are
  natural next experiments.
- The training call does not yet provide padding or causal attention masks to
  the model; padded loss values are masked correctly.
- Checkpoint persistence and MIDI export are not yet implemented.

## Verification

```bash
python -m compileall -q .
python train.py
```

Both commands were verified with TensorFlow 2.21 / Keras 3. This project is
based on the Transformer melody-generation material from
[The Sound of AI](https://www.youtube.com/@ValerioVelardoTheSoundofAI).
