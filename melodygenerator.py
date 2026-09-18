"""Seeded top-k/temperature decoding and MIDI export of pitch-duration tokens."""

import re
from pathlib import Path
import numpy as np
import tensorflow as tf


class MelodyGenerator:
    def __init__(self, transformer, tokenizer, max_length=32):
        self.transformer, self.tokenizer, self.max_length = transformer, tokenizer, max_length

    def generate(self, start_sequence, temperature=0.8, top_k=5, seed=7):
        if not start_sequence or temperature < 0 or top_k < 1:
            raise ValueError("A nonempty known seed, nonnegative temperature and positive top_k are required")
        unknown = set(start_sequence) - set(self.tokenizer.word_index)
        if unknown:
            raise ValueError(f"Unknown seed notes: {sorted(unknown)}")
        max_positions = min(self.transformer.model_config["max_num_positions_in_pe_encoder"],
                            self.transformer.model_config["max_num_positions_in_pe_decoder"])
        if not len(start_sequence) <= self.max_length <= max_positions:
            raise ValueError("Requested length must include the seed and fit positional encodings")
        ids = self.tokenizer.texts_to_sequences([start_sequence])[0]
        rng = np.random.default_rng(seed)
        while len(ids) < self.max_length:
            logits = self.transformer(tf.constant([ids]), training=False).numpy()[0, -1].copy()
            logits[0] = -np.inf
            if temperature == 0:
                next_id = int(np.argmax(logits))
            else:
                candidates = np.argsort(logits)[-min(top_k, len(logits) - 1):]
                scores = logits[candidates] / temperature
                probabilities = np.exp(scores - scores.max())
                next_id = int(rng.choice(candidates, p=probabilities / probabilities.sum()))
            ids.append(next_id)
        return " ".join(self.tokenizer.index_word[i] for i in ids)


def write_midi(tokens, path, bpm=100):
    import mido
    if bpm <= 0:
        raise ValueError("Tempo must be positive")
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    pitch_classes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    for token in tokens:
        match = re.fullmatch(r"([A-G])([b#]?)(-?\d+)-(\d+(?:\.\d+)?)", token)
        if not match:
            raise ValueError(f"Unsupported note token: {token}")
        letter, accidental, octave, duration = match.groups()
        pitch = 12 * (int(octave) + 1) + pitch_classes[letter] + {"": 0, "b": -1, "#": 1}[accidental]
        ticks = round(float(duration) * midi.ticks_per_beat)
        if not 0 <= pitch <= 127 or ticks <= 0:
            raise ValueError(f"Out-of-range note or duration: {token}")
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=ticks))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
