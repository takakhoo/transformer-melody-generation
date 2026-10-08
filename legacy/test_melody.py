import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from melodygenerator import MelodyGenerator, write_midi
from melodypreprocessor import MelodyPreprocessor
from train import _calculate_loss, load_artifact
from transformer import Transformer, sinusoidal_position_encoding


class MelodyTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(7)
        self.model = Transformer(1, 16, 2, 32, 8, 8, 20, 20, dropout_rate=0)

    def test_no_future_leak_through_either_stack(self):
        first = tf.constant([[1, 2, 3, 4, 5]])
        changed = tf.constant([[1, 2, 3, 7, 6]])
        np.testing.assert_allclose(self.model(first)[:, :3], self.model(changed)[:, :3], atol=1e-6)
        np.testing.assert_allclose(self.model(first)[:, :3], self.model(first[:, :3]), atol=1e-6)

    def test_padding_does_not_change_prefix_and_loss_is_safe(self):
        prefix = tf.constant([[1, 2, 3]])
        padded = tf.constant([[1, 2, 3, 0, 0]])
        np.testing.assert_allclose(self.model(prefix), self.model(padded)[:, :3], atol=1e-6)
        self.assertEqual(float(_calculate_loss(tf.zeros((1, 3), tf.int32), self.model(prefix))), 0)

    def test_target_is_shifted_exactly_once(self):
        p = MelodyPreprocessor("unused")
        x, y = p._create_sequence_pairs([[1, 2, 3, 4], [5, 6]])
        np.testing.assert_array_equal(x, [[1, 2, 3], [5, 0, 0]])
        np.testing.assert_array_equal(y, [[2, 3, 4], [6, 0, 0]])

    def test_positional_encoding_interleaves_and_supports_odd_width(self):
        encoding = sinusoidal_position_encoding(3, 5).numpy()[0]
        np.testing.assert_allclose(encoding[0], [0, 1, 0, 1, 0])
        np.testing.assert_allclose(encoding[1, :2], [np.sin(1), np.cos(1)], atol=1e-7)

    def test_seeded_generation_and_validation(self):
        p = MelodyPreprocessor("unused")
        p.tokenizer.fit_on_texts([["C4-1.0", "D4-0.5", "E4-2.0", "F4-1.0", "G4-1.0", "A4-1.0", "B4-1.0"]])
        g = MelodyGenerator(self.model, p.tokenizer, max_length=8)
        first = g.generate(["C4-1.0"], seed=19)
        self.assertEqual(first, g.generate(["C4-1.0"], seed=19))
        self.assertEqual(len(first.split()), 8)
        with self.assertRaises(ValueError):
            g.generate(["unknown"])
        with self.assertRaises(ValueError):
            g.generate([])

    def test_checkpoint_roundtrip(self):
        import json
        p = MelodyPreprocessor("unused")
        p.tokenizer.fit_on_texts([["C4-1.0"]])
        x = tf.constant([[1, 2, 3]])
        original = self.model(x).numpy()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "model_config.json").write_text(json.dumps(self.model.model_config))
            (directory / "tokenizer.json").write_text(p.tokenizer.to_json())
            self.model.save_weights(directory / "model.weights.h5")
            restored, _ = load_artifact(directory)
            np.testing.assert_array_equal(original, restored(x).numpy())

    def test_midi_pitch_duration_and_tempo(self):
        import mido
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.mid"
            write_midi(["C4-1.0", "Bb4-0.5", "F#4-2.0"], path)
            midi = mido.MidiFile(path)
            self.assertEqual([m.note for m in midi.tracks[0] if m.type == "note_on"], [60, 70, 66])
            self.assertEqual([m.time for m in midi.tracks[0] if m.type == "note_off"], [480, 240, 960])
            self.assertAlmostEqual(midi.length, 2.1)


if __name__ == "__main__":
    unittest.main()
