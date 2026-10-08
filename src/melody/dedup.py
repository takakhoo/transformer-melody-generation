"""Transposition-invariant near-duplicate search: shingle the interval/rhythm symbol sequence,
MinHash, band with LSH, then verify candidates exactly.

Verification reports two scores per pair:
  jaccard  - overlap of k-shingle sets (order-free, robust to ornaments and local edits)
  lcs_frac - longest common contiguous run of symbols, as a fraction of the shorter melody
"""
from collections import defaultdict

import numpy as np

MERSENNE = (1 << 61) - 1


def shingles(seq, k):
    """Set of hashed k-grams of a 1-D or 2-D integer symbol sequence."""
    a = np.asarray(seq)
    if a.ndim == 2:
        a = a[:, 0] * 64 + a[:, 1]
    if len(a) < k:
        return np.array([], dtype=np.uint64)
    view = np.lib.stride_tricks.sliding_window_view(a.astype(np.int64), k)
    h = np.zeros(len(view), dtype=np.uint64)
    for j in range(k):
        h = h * np.uint64(1000003) ^ (view[:, j].astype(np.uint64) + np.uint64(97))
    return np.unique(h)


class MinHashLSH:
    def __init__(self, n_perm=128, bands=32, seed=0):
        assert n_perm % bands == 0
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE, n_perm, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, n_perm, dtype=np.uint64)
        self.bands, self.rows = bands, n_perm // bands

    def signature(self, sh):
        if len(sh) == 0:
            return None
        x = (sh[:, None] % np.uint64(MERSENNE))
        hv = (self.a[None] * x + self.b[None]) % np.uint64(MERSENNE)
        return hv.min(0)

    def candidates(self, sigs):
        """sigs: dict id -> signature. Returns the set of candidate id pairs sharing a band."""
        buckets = defaultdict(list)
        for key, s in sigs.items():
            if s is None:
                continue
            for b in range(self.bands):
                buckets[(b, s[b * self.rows:(b + 1) * self.rows].tobytes())].append(key)
        pairs = set()
        for ids in buckets.values():
            if 1 < len(ids) < 500:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pairs.add((ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i]))
        return pairs


def jaccard(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    inter = len(np.intersect1d(s1, s2, assume_unique=True))
    return inter / (len(s1) + len(s2) - inter)


def longest_common_run(x, y):
    """Length of the longest common contiguous subsequence of two symbol arrays (O(nm), small inputs)."""
    a, b = np.asarray(x), np.asarray(y)
    if a.ndim == 2:
        a = a[:, 0] * 64 + a[:, 1]
        b = b[:, 0] * 64 + b[:, 1]
    if len(a) == 0 or len(b) == 0:
        return 0
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    best = 0
    for i in range(len(a)):
        cur = np.zeros(len(b) + 1, dtype=np.int32)
        eq = a[i] == b
        cur[1:][eq] = prev[:-1][eq] + 1
        best = max(best, int(cur.max()))
        prev = cur
    return best


def verify(seq_a, seq_b, sh_a, sh_b):
    run = longest_common_run(seq_a, seq_b)
    return {"jaccard": jaccard(sh_a, sh_b), "lcs": run, "lcs_frac": run / max(1, min(len(seq_a), len(seq_b)))}
