"""Explicit note vocabulary and once-shifted, right-padded training pairs."""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


class MelodyPreprocessor:
    def __init__(self, dataset_path, batch_size=32, seed=7):
        self.dataset_path, self.batch_size, self.seed = dataset_path, batch_size, seed
        self.tokenizer = Tokenizer(filters="", lower=False, split=",")
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        return self.number_of_tokens + 1

    def _load_dataset(self):
        with open(self.dataset_path) as handle:
            return json.load(handle)

    def _parse_melody(self, melody):
        tokens = [token.strip() for token in melody.split(",")]
        if len(tokens) < 2 or any(not token for token in tokens):
            raise ValueError("Each melody needs at least two nonempty note tokens")
        return tokens

    def _create_sequence_pairs(self, melodies):
        # One example per melody. Target shift occurs here and nowhere else.
        width = max(len(melody) - 1 for melody in melodies)
        inputs = np.zeros((len(melodies), width), dtype=np.int32)
        targets = np.zeros_like(inputs)
        for i, melody in enumerate(melodies):
            inputs[i, :len(melody) - 1] = melody[:-1]
            targets[i, :len(melody) - 1] = melody[1:]
        return inputs, targets

    def create_training_dataset(self):
        melodies = [self._parse_melody(m) for m in self._load_dataset()]
        self.tokenizer.fit_on_texts(melodies)
        encoded = self.tokenizer.texts_to_sequences(melodies)
        self.number_of_tokens = len(self.tokenizer.word_index)
        self.max_melody_length = max(map(len, encoded))
        pairs = self._create_sequence_pairs(encoded)
        return tf.data.Dataset.from_tensor_slices(pairs).shuffle(
            len(encoded), seed=self.seed
        ).batch(self.batch_size)
