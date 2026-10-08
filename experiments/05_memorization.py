"""Memorization in melody language models as a function of model size, duplication, and data
deduplication, on PDMX (public domain / CC0).

Design
- Near-duplicate families from the transposition-invariant audit; 3% of families held out whole
  for testing, so no test melody has a twin in training.
- RAW corpus: every melody in the training families (natural duplication kept).
  DEDUP corpus: one melody per family. Both get the same canaries and the same token budget.
- Canaries: synthetic melodies inserted 1, 2, 4, 8, 16 or 32 times; extraction is greedy
  continuation of a 12-note prompt; exposure compares canary NLL with never-inserted twins.
- Natural memorization: extraction of real training melodies by family size.
- Copying in free samples: longest transposed run shared with any training melody."""
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from melody import dedup, sources, tokens as T  # noqa: E402
from melody.representation import Melody, interval_rhythm_tokens  # noqa: E402

OUT = ROOT / "results" / "memorization"
OUT.mkdir(parents=True, exist_ok=True)
DUPS = (1, 2, 4, 8, 16, 32)
PER_DUP = 60
PROMPT_NOTES, CONT_NOTES = 12, 20


def families():
    path = OUT / "families.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    import importlib
    audit = importlib.import_module("03_corpus_audit")
    mels = [m for m in sources.pdmx() if 32 <= len(m) <= 400]
    keep, edges = audit.graph(mels)
    parent = list(range(len(mels)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    fam = np.array([find(i) for i in range(len(mels))])
    res = {"mels": mels, "family": fam, "n_edges": len(edges)}
    path.write_bytes(pickle.dumps(res))
    return res


def canaries(rng, n):
    """Plausible but novel melodies: scale-constrained random walks with common rhythm cells."""
    scale = np.array([0, 2, 4, 5, 7, 9, 11])
    cells = [[1, 1], [0.5, 0.5, 1], [1.5, 0.5], [0.5, 0.5, 0.5, 0.5], [2], [1, 0.5, 0.5]]
    out = []
    for _ in range(n):
        tonic = int(rng.integers(55, 67))
        deg, pitch, dur = int(rng.integers(0, 7)), [], []
        while len(pitch) < 48:
            for d in cells[int(rng.integers(len(cells)))]:
                deg = int(np.clip(deg + rng.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3]), -7, 13))
                pitch.append(tonic + 12 * (deg // 7) + scale[deg % 7])
                dur.append(d)
        dur = np.array(dur[:48], float)
        out.append(Melody(np.array(pitch[:48]), np.concatenate([[0], np.cumsum(dur)[:-1]]), dur, "canary"))
    return out


def build_corpora(seed=0):
    rng = np.random.default_rng(seed)
    F = families()
    mels, fam = F["mels"], F["family"]
    uniq = np.unique(fam)
    test_fams = set(rng.choice(uniq, size=int(0.03 * len(uniq)), replace=False).tolist())
    test = [m for m, f in zip(mels, fam) if f in test_fams]
    frac = float(os.environ.get("TRAIN_FRAC", "0.15"))
    train_fams = set(rng.choice([f for f in uniq if f not in test_fams], size=int(frac * (len(uniq) - len(test_fams))), replace=False).tolist())
    train_idx = [i for i, f in enumerate(fam) if f in train_fams]
    seen, dedup_idx = set(), []
    for i in rng.permutation(train_idx):
        if fam[i] not in seen:
            seen.add(fam[i])
            dedup_idx.append(i)
    fam_size = {}
    for i in train_idx:
        fam_size[fam[i]] = fam_size.get(fam[i], 0) + 1
    can = canaries(rng, PER_DUP * len(DUPS) + PER_DUP)
    inserted = {d: can[k * PER_DUP:(k + 1) * PER_DUP] for k, d in enumerate(DUPS)}
    held_canaries = can[len(DUPS) * PER_DUP:]
    extra = [m for d, ms in inserted.items() for m in ms for _ in range(d)]
    raw = [mels[i] for i in train_idx] + extra
    ded = [mels[i] for i in dedup_idx] + extra
    return {"raw": raw, "dedup": ded, "test": test, "inserted": inserted, "held_canaries": held_canaries,
            "train_mels": [mels[i] for i in train_idx], "train_fam_size": [fam_size[fam[i]] for i in train_idx]}


def kgram_index(mels, k=12):
    idx = set()
    for m in mels:
        idx.update(dedup.shingles(interval_rhythm_tokens(m), k).tolist())
    return idx


def longest_copied_run(m, index, k=12):
    """Longest run of consecutive interval/rhythm k-grams of m that all occur in the training index,
    converted to notes (run of r k-grams spans r + k + 1 notes)."""
    toks = interval_rhythm_tokens(m)
    if len(toks) < k:
        return 0
    a = toks[:, 0] * 64 + toks[:, 1]
    view = np.lib.stride_tricks.sliding_window_view(a.astype(np.int64), k)
    h = np.zeros(len(view), dtype=np.uint64)
    for j in range(k):
        h = h * np.uint64(1000003) ^ (view[:, j].astype(np.uint64) + np.uint64(97))
    hit = np.array([int(x) in index for x in h])
    best = run = 0
    for v in hit:
        run = run + 1 if v else 0
        best = max(best, run)
    return best + k + 1 if best else 0


def extraction(model, mels, train_lm):
    """Greedy continuation of the first PROMPT_NOTES notes; success if the next CONT_NOTES notes
    match exactly (pitch and duration tokens). Batched: all prompts have the same length."""
    need = 1 + 2 * (PROMPT_NOTES + CONT_NOTES)
    toks = [T.encode(m) for m in mels]
    toks = [t for t in toks if len(t) >= need]
    if not toks:
        return float("nan"), 0
    L = 1 + 2 * PROMPT_NOTES
    gens = train_lm.generate_batch(model, [t[:L] for t in toks], 2 * CONT_NOTES, greedy=True)
    ok = [np.array_equal(g[L:L + 2 * CONT_NOTES], t[L:L + 2 * CONT_NOTES]) for g, t in zip(gens, toks)]
    return float(np.mean(ok)), len(ok)


def run(size, corpus, token_budget, seed=0):
    from melody import train_lm
    augment = os.environ.get("AUGMENT", "1") == "1"
    frac = os.environ.get("TRAIN_FRAC", "0.15")
    tag = f"{corpus}_{size}" + ("" if augment else "_noaug") + ("" if float(frac) == 0.15 else f"_frac{frac}")
    path = OUT / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    stamp = time.time()

    def note(msg):
        print(f"[{tag}] {msg} (+{time.time() - stamp:.0f}s)", flush=True)
    C = build_corpora(seed)
    note(f"corpora built: {len(C[corpus])} training melodies, {len(C['test'])} test")
    rng = np.random.default_rng(seed + 1)
    seqs = [T.encode(m, max_notes=512) for m in C[corpus]]

    def epoch_seqs(epoch):  # random transposition each epoch, the usual augmentation (unless AUGMENT=0)
        r = np.random.default_rng(seed * 100 + epoch)
        return [T.encode(m, transpose=int(r.integers(-5, 7)) if augment else 0, max_notes=512) for m in C[corpus]]
    t0 = time.time()
    model = train_lm.train(epoch_seqs, size=size, token_budget=token_budget, seed=seed,
                           log=lambda s: print(f"[{tag}] {s}", flush=True))
    note("trained")
    test_nll = train_lm.nll_per_note(model, [T.encode(m, max_notes=512) for m in C["test"][:1500]])
    note(f"test NLL {test_nll:.3f}")
    res = {"size": size, "corpus": corpus, "augment": augment, "train_frac": float(frac), "params": model.n_params(), "tokens_seen": token_budget,
           "train_melodies": len(C[corpus]), "test_nll_per_note": test_nll, "minutes": (time.time() - t0) / 60}
    for d, ms in C["inserted"].items():
        res[f"canary_extract_x{d}"] = extraction(model, ms, train_lm)[0]
        res[f"canary_nll_x{d}"] = train_lm.nll_per_note(model, [T.encode(m) for m in ms])
    res["canary_nll_x0"] = train_lm.nll_per_note(model, [T.encode(m) for m in C["held_canaries"]])
    note("canaries done")
    sizes = np.array(C["train_fam_size"])
    for lo, hi, name in ((1, 1, "fam1"), (2, 3, "fam2-3"), (4, 9, "fam4-9"), (10, 10**6, "fam10+")):
        pick = [m for m, s in zip(C["train_mels"], sizes) if lo <= s <= hi]
        pick = [pick[i] for i in rng.permutation(len(pick))[:150]]
        res[f"natural_extract_{name}"] = extraction(model, pick, train_lm)[0]
    note("natural extraction done")
    index = kgram_index(C["train_mels"])
    note(f"k-gram index: {len(index):,} shingles")
    runs = {}
    for temp in (0.8, 1.0):
        gens = train_lm.generate_batch(model, [[T.BOS]] * 300, 2 * 64, temperature=temp, seed=int(temp * 100))
        lens = np.array([longest_copied_run(T.to_melody(g), index) for g in gens])
        runs[str(temp)] = {"median_run": float(np.median(lens)), "frac_run>=20": float(np.mean(lens >= 20)),
                           "frac_run>=32": float(np.mean(lens >= 32))}
    res["sample_copying"] = runs
    note("samples done")
    path.write_text(json.dumps(res, indent=2))
    import torch
    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    return res


if __name__ == "__main__":
    budget = int(float(os.environ.get("TOKENS", "40e6")))
    sizes = sys.argv[1:] or ["XS", "S", "M", "L"]
    for size in sizes:
        for corpus in os.environ.get("CORPORA", "dedup,raw").split(","):
            r = run(size, corpus, budget)
            print(f"{corpus:5s} {size}: params {r['params']:,} test NLL/note {r['test_nll_per_note']:.3f} | canary extract "
                  + " ".join(f"x{d}:{r[f'canary_extract_x{d}']:.2f}" for d in DUPS)
                  + f" | natural fam1 {r['natural_extract_fam1']:.2f} fam10+ {r['natural_extract_fam10+']:.2f}"
                  + f" | samples run>=20 @T1.0 {r['sample_copying']['1.0']['frac_run>=20']:.2f}", flush=True)
