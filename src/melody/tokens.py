"""Pitch/duration tokens: BOS (P D)* EOS, durations snapped to a small set of musical values."""
import numpy as np

from .representation import REST, Melody

PAD, BOS, EOS = 0, 1, 2
PITCH_LO, PITCH_HI = 36, 96                       # C2..C7; out-of-range notes fold by octaves
DUR_TICKS = np.array([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96])  # 12 ticks = 1 beat
P0 = 3
N_PITCH = PITCH_HI - PITCH_LO + 1
REST_TOK = P0 + N_PITCH
D0 = REST_TOK + 1
VOCAB = D0 + len(DUR_TICKS)


def _fold(p):
    while p < PITCH_LO:
        p += 12
    while p > PITCH_HI:
        p -= 12
    return p


LOG_DUR = np.log(DUR_TICKS)
_MID = np.exp((LOG_DUR[1:] + LOG_DUR[:-1]) / 2)  # geometric midpoints for nearest-in-log snapping


def encode(m: Melody, transpose=0, max_notes=None):
    p, d = m.quantized()
    if max_notes:
        p, d = p[:max_notes], d[:max_notes]
    rest = p == REST
    q = p + transpose
    q = np.where(q < PITCH_LO, q + 12 * np.ceil((PITCH_LO - q) / 12), q)
    q = np.where(q > PITCH_HI, q - 12 * np.ceil((q - PITCH_HI) / 12), q).astype(np.int64)
    ptok = np.where(rest, REST_TOK, P0 + q - PITCH_LO)
    dtok = D0 + np.searchsorted(_MID, np.maximum(1, d))
    out = np.empty(2 * len(p) + 2, np.int16)
    out[0], out[-1] = BOS, EOS
    out[1:-1:2], out[2:-1:2] = ptok, dtok
    return out


def decode(tokens):
    """Tokens -> (pitch array with REST, duration in beats)."""
    p, d = [], []
    toks = [t for t in tokens if t not in (PAD, BOS)]
    for i in range(0, len(toks) - 1, 2):
        a, b = toks[i], toks[i + 1]
        if a == EOS or b == EOS or not (P0 <= a <= REST_TOK) or not (D0 <= b < VOCAB):
            break
        p.append(REST if a == REST_TOK else a - P0 + PITCH_LO)
        d.append(DUR_TICKS[b - D0] / 12)
    return np.array(p, int), np.array(d, float)


def to_melody(tokens, source="generated"):
    p, d = decode(tokens)
    onset = np.concatenate([[0.0], np.cumsum(d)[:-1]]) if len(d) else np.array([])
    return Melody(p, onset, d, source)
