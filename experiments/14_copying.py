"""Copying in free samples, measured strictly: the longest passage of a generated melody that matches
one training melody contiguously, up to transposition and tempo. Experiment 05 reports a looser
statistic (consecutive k-grams that each occur somewhere in training, possibly in different melodies);
this script reports both, plus a filter that drops trivial passages (repeated notes, a single interval).
It also writes the listening examples for the site.

Usage: python experiments/14_copying.py [tag ...]   (default: every model in results/memorization)"""
import importlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from melody import tokens as T, train_lm  # noqa: E402
from melody.models import Transformer  # noqa: E402
from melody.representation import interval_rhythm_tokens  # noqa: E402

K = 12
CAP = 200
N_SAMPLES = 300
mem = importlib.import_module("05_memorization")


def kgram_hashes(toks):
    if len(toks) < K:
        return np.array([], np.uint64)
    a = (toks[:, 0] * 64 + toks[:, 1]).astype(np.int64)
    view = np.lib.stride_tricks.sliding_window_view(a, K)
    h = np.zeros(len(view), dtype=np.uint64)
    for j in range(K):
        h = h * np.uint64(1000003) ^ (view[:, j].astype(np.uint64) + np.uint64(97))
    return h


def build_index(train):
    index = {}
    for i, m in enumerate(train):
        for j, h in enumerate(kgram_hashes(interval_rhythm_tokens(m)).tolist()):
            lst = index.setdefault(h, [])
            if len(lst) < CAP:
                lst.append((i, j))
    return index


def periodic(sym, max_period=8, agree=0.9):
    """True if the passage is a short cell repeated (an ostinato, trill or sequence loop)."""
    a = sym[:, 0] * 64 + sym[:, 1]
    for p in range(1, max_period + 1):
        if len(a) >= 2 * p and np.mean(a[p:] == a[:-p]) >= agree:
            return True
    return False


def nontrivial(sym):
    """A melodic passage: at least four distinct intervals, at most half repeated notes, and not a
    short cell looped. Repeated notes, trills, ostinati and scale loops occur in thousands of scores."""
    iv = sym[:, 0]
    return len(np.unique(iv)) >= 4 and np.mean(iv == 0) <= 0.5 and not periodic(sym)


def strict_copy(m, index):
    """Longest contiguous match with a single training melody, overall and among non-trivial passages.
    Returns notes copied, the source (melody index, first source note) and the first sample note."""
    sym = interval_rhythm_tokens(m)
    hs = kgram_hashes(sym).tolist()
    prev, loose, run = {}, 0, 0
    best = {"strict": 0, "nt": 0}
    where = {"strict": None, "nt": None}

    def close(i, j, k, r):  # a run of r k-grams ending at sample k-gram k and source k-gram j
        if r > best["strict"]:
            best["strict"], where["strict"] = r, (i, j, k)
        if r > best["nt"] and nontrivial(sym[k - r + 1:k + K]):
            best["nt"], where["nt"] = r, (i, j, k)
    for k, h in enumerate(hs):
        hits = index.get(h)
        run = run + 1 if hits else 0
        loose = max(loose, run)
        cur = {}
        for i, j in hits or ():
            cur[(i, j)] = prev.get((i, j - 1), 0) + 1
        for (i, j), r in prev.items():
            if (i, j + 1) not in cur:
                close(i, j, k - 1, r)
        prev = cur
    for (i, j), r in prev.items():
        close(i, j, len(hs) - 1, r)
    out = {"loose": loose + K + 1 if loose else 0}
    for key in ("strict", "nt"):
        r = best[key]
        out[key] = r + K + 1 if r else 0
        if r:
            i, j, k = where[key]
            out[f"{key}_src"], out[f"{key}_src_note"], out[f"{key}_sample_note"] = int(i), int(j - r + 1), int(k - r + 1)
    return out


def notes(m, lo=0, hi=None):
    p, o, d = m.notes()
    p, o, d = p[lo:hi], o[lo:hi], d[lo:hi]
    o = o - (o[0] if len(o) else 0)
    return [[int(a), round(float(b), 3), round(float(c), 3)] for a, b, c in zip(p, o, d)]


def load(tag):
    size = tag.split("_")[1]
    model = Transformer("tokens", vocab=T.VOCAB, **train_lm.SIZES[size], dropout=0.0)
    model.load_state_dict(torch.load(mem.OUT / f"{tag}.pt", map_location="cpu"))
    return model.to(train_lm.DEV).eval()


if __name__ == "__main__":
    tags = sys.argv[1:] or sorted(p.stem for p in mem.OUT.glob("*.pt"))
    pools = {}
    out_path = ROOT / "results" / "14_copying.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    site = {}
    for tag in tags:
        frac = next((x[4:] for x in tag.split("_") if x.startswith("frac")), "0.15")
        os.environ["TRAIN_FRAC"] = frac
        if frac not in pools:
            C = mem.build_corpora(0)
            pools[frac] = (C["train_mels"], build_index(C["train_mels"]))
        train, index = pools[frac]
        model = load(tag)
        res = {}
        for temp in (0.8, 1.0):
            gens = train_lm.generate_batch(model, [[T.BOS]] * N_SAMPLES, 2 * 64, temperature=temp, seed=int(temp * 100))
            mels = [T.to_melody(g) for g in gens]
            mels = [m for m in mels if len(m.notes()[0]) >= 24]
            cs = [strict_copy(m, index) for m in mels]
            strict = np.array([c["strict"] for c in cs])
            loose = np.array([c["loose"] for c in cs])
            nt = np.array([c["nt"] for c in cs])
            res[str(temp)] = {"n": len(cs), "loose_run>=20": float(np.mean(loose >= 20)), "strict_run>=20": float(np.mean(strict >= 20)),
                              "strict_nontrivial_run>=20": float(np.mean(nt >= 20)), "strict_nontrivial_run>=32": float(np.mean(nt >= 32)),
                              "median_strict": float(np.median(strict)), "median_nontrivial": float(np.median(nt)),
                              "max_nontrivial": int(nt.max()) if len(nt) else 0}
            if temp == 0.8:
                ranked = sorted(zip(mels, cs), key=lambda x: -x[1]["nt"])
                copies = []
                for m, c in ranked[:8]:
                    if not c["nt"]:
                        continue
                    src = train[c["nt_src"]]
                    a, b, n = c["nt_sample_note"], c["nt_src_note"], c["nt"]
                    copies.append({"copied_notes": n, "title": str(src.meta.get("title", ""))[:80], "sample": notes(m, 0, 48),
                                   "sample_seg": notes(m, a, a + n), "source": notes(src, b, b + n),
                                   "transpose": int(np.median(m.notes()[0][a:a + n]) - np.median(src.notes()[0][b:b + n]))})
                originals = [{"sample": notes(m, 0, 48), "copied_notes": 0} for m, c in zip(mels, cs) if c["strict"] == 0][:8]
                site[tag] = {"n": len(cs), "share_nt14": float(np.mean(nt > 0)), "share_nt20": res["0.8"]["strict_nontrivial_run>=20"],
                             "share_trivial20": res["0.8"]["strict_run>=20"], "copies": copies, "originals": originals}
        results[tag] = res
        print(tag, json.dumps(res), flush=True)
        out_path.write_text(json.dumps(results, indent=1))
    keep = next(([f"raw_{z}", f"dedup_{z}"] for z in ("M", "S") if f"raw_{z}" in site and f"dedup_{z}" in site), None)
    if keep:  # the site page compares a raw-trained and a dedup-trained model of the same size
        (ROOT / "site" / "data" / "samples.json").write_text(json.dumps({t: site[t] for t in keep}, separators=(",", ":")))
