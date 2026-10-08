"""A melody is three parallel arrays: MIDI pitch (or -1 for a rest), onset and duration in
quarter-note beats. Everything downstream (tokens, fingerprints) derives from that."""
from dataclasses import dataclass, field

import numpy as np

REST = -1
DUR_GRID = 12  # ticks per quarter note: covers triplets and sixteenths exactly


@dataclass
class Melody:
    pitch: np.ndarray
    onset: np.ndarray
    duration: np.ndarray
    source: str = ""
    piece_id: str = ""
    split: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.pitch)

    def notes(self):
        """Sounding notes only (rests removed), keeping their onsets."""
        keep = self.pitch != REST
        return self.pitch[keep], self.onset[keep], self.duration[keep]

    def quantized(self):
        """Durations on the 1/12-beat grid, with rests made explicit from onset gaps (vectorized)."""
        p, o, d = self.notes()
        if len(p) == 0:
            return np.array([], int), np.array([], int)
        on = np.round(o * DUR_GRID).astype(np.int64)
        du = np.maximum(1, np.round(d * DUR_GRID).astype(np.int64))
        nxt = np.append(on[1:], on[-1] + du[-1])
        dur = np.maximum(1, np.minimum(du, nxt - on))
        gap = nxt - (on + dur)
        has = gap > 0
        n = len(p) + int(has.sum())
        pos = np.arange(len(p)) + np.concatenate([[0], np.cumsum(has)[:-1]])
        out_p = np.empty(n, np.int64)
        out_d = np.empty(n, np.int64)
        out_p[pos], out_d[pos] = p, dur
        rest_pos = pos[has] + 1
        out_p[rest_pos], out_d[rest_pos] = REST, gap[has]
        return out_p, out_d


def interval_rhythm_tokens(m: Melody, ratio_bins=(0.26, 0.41, 0.59, 0.84, 1.19, 1.68, 2.38, 3.36)):
    """Transposition- and tempo-invariant symbols: (pitch interval to the previous sounding note,
    quantized log ratio of inter-onset intervals). Rests are folded into the IOI."""
    p, o, _ = m.notes()
    if len(p) < 3:
        return np.zeros((0, 2), int)
    ioi = np.diff(o)
    ioi = np.where(ioi <= 0, 1e-3, ioi)
    iv = np.clip(np.diff(p), -24, 24)
    ratio = ioi[1:] / ioi[:-1]
    rb = np.digitize(ratio, ratio_bins)
    return np.column_stack([iv[1:], rb])


def frame_tokens(top_line):
    """For piano-roll benchmarks: the top line as a frame sequence, made transposition invariant
    by taking frame-to-frame intervals (0 while a pitch is held, 99 for silence boundaries)."""
    x = np.asarray(top_line, float)
    out = np.empty(len(x) - 1, int)
    for i in range(1, len(x)):
        a, b = x[i - 1], x[i]
        if np.isnan(a) or np.isnan(b):
            out[i - 1] = 99 if np.isnan(a) != np.isnan(b) else 98
        else:
            out[i - 1] = int(np.clip(b - a, -48, 48))
    return out
