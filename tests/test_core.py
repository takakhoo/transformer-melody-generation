import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from melody import dedup, tokens as T  # noqa: E402
from melody.representation import Melody, interval_rhythm_tokens  # noqa: E402


def tune(seed=0, n=40, tonic=60):
    rng = np.random.default_rng(seed)
    pitch = tonic + np.cumsum(rng.choice([-2, -1, 0, 1, 2, 3], n))
    dur = rng.choice([0.5, 1.0, 1.5, 2.0], n)
    return Melody(pitch, np.concatenate([[0], np.cumsum(dur)[:-1]]), dur, "test")


class Fingerprint(unittest.TestCase):
    def test_transposition_and_tempo_invariant(self):
        m = tune()
        moved = Melody(m.pitch + 7, m.onset * 2, m.duration * 2, "test")
        np.testing.assert_array_equal(interval_rhythm_tokens(m), interval_rhythm_tokens(moved))

    def test_rest_folded_into_ioi(self):
        m = tune()
        with_rest = Melody(np.insert(m.pitch, 5, -1), np.insert(m.onset, 5, m.onset[5] - 0.25),
                           np.insert(m.duration, 5, 0.25), "test")
        np.testing.assert_array_equal(interval_rhythm_tokens(m), interval_rhythm_tokens(with_rest))

    def test_jaccard_and_run(self):
        a = interval_rhythm_tokens(tune(0))
        b = interval_rhythm_tokens(tune(1))
        sa, sb = dedup.shingles(a, 8), dedup.shingles(b, 8)
        self.assertEqual(dedup.jaccard(sa, sa), 1.0)
        self.assertLess(dedup.jaccard(sa, sb), 0.2)
        self.assertEqual(dedup.longest_common_run(a, a), len(a))

    def test_lsh_finds_transposed_copy(self):
        mels = [tune(s) for s in range(30)] + [Melody(tune(3).pitch - 5, tune(3).onset, tune(3).duration, "x")]
        sh = {i: dedup.shingles(interval_rhythm_tokens(m), 8) for i, m in enumerate(mels)}
        lsh = dedup.MinHashLSH()
        pairs = lsh.candidates({i: lsh.signature(s) for i, s in sh.items()})
        self.assertIn((3, 30), pairs)


class CopyFilter(unittest.TestCase):
    def test_ostinato_is_trivial_and_tune_is_not(self):
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
        cp = importlib.import_module("14_copying")
        loop = np.array([[2, 4], [1, 4], [-1, 4], [-2, 4]] * 6)
        self.assertTrue(cp.periodic(loop))
        self.assertFalse(cp.nontrivial(loop))
        self.assertTrue(cp.nontrivial(interval_rhythm_tokens(tune(5))[:20]))


class Tokens(unittest.TestCase):
    def test_roundtrip(self):
        m = tune()
        back = T.to_melody(T.encode(m))
        np.testing.assert_array_equal(back.notes()[0], m.pitch)
        np.testing.assert_allclose(back.notes()[2], m.duration)

    def test_transpose(self):
        m = tune()
        np.testing.assert_array_equal(T.to_melody(T.encode(m, transpose=2)).notes()[0], m.pitch + 2)


class Model(unittest.TestCase):
    def test_causal(self):
        import torch
        from melody.models import Transformer
        torch.manual_seed(0)
        model = Transformer("tokens", vocab=T.VOCAB, d=32, layers=2, heads=2, dropout=0.0).eval()
        x = torch.randint(3, T.VOCAB, (1, 20))
        y = x.clone()
        y[0, 12:] = 3
        with torch.no_grad():
            a, b = model(x), model(y)
        torch.testing.assert_close(a[0, :12], b[0, :12])


if __name__ == "__main__":
    unittest.main()
