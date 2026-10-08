"""Transposition-invariant near-duplicate audit of melody corpora.

For each corpus: build the near-duplicate graph (MinHash LSH over 8-symbol interval/rhythm
shingles, then exact verification), report how many melodies have a twin, how test pieces of the
official split relate to training pieces, and the leakage a random 80/10/10 split would produce."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melody import dedup, sources  # noqa: E402
from melody.representation import interval_rhythm_tokens  # noqa: E402

K = 8
MIN_SYMBOLS = 16
TWIN = {"jaccard": 0.5, "lcs": 12}     # near-duplicate: half the phrases shared or 12+ symbols in a row
COPY = {"jaccard": 0.9}                 # transposed copy


def graph(mels):
    toks = [interval_rhythm_tokens(m) for m in mels]
    keep = [i for i, t in enumerate(toks) if len(t) >= MIN_SYMBOLS]
    sh = {i: dedup.shingles(toks[i], K) for i in keep}
    lsh = dedup.MinHashLSH(n_perm=128, bands=32)
    sigs = {i: lsh.signature(sh[i]) for i in keep}
    edges = {}
    for a, b in lsh.candidates(sigs):
        j = dedup.jaccard(sh[a], sh[b])
        if j < 0.15:  # the exact longest-run check only for pairs that share phrases
            continue
        v = dedup.verify(toks[a], toks[b], sh[a], sh[b])
        if v["jaccard"] >= TWIN["jaccard"] or v["lcs"] >= TWIN["lcs"]:
            edges[(a, b)] = v
    return keep, edges


def components(n_nodes, edges):
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in range(n_nodes):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def audit(name, mels, rng):
    keep, edges = graph(mels)
    nbrs = defaultdict(list)
    for (a, b), v in edges.items():
        nbrs[a].append((b, v))
        nbrs[b].append((a, v))
    n = len(keep)
    has_twin = sum(1 for i in keep if nbrs[i])
    has_copy = sum(1 for i in keep if any(v["jaccard"] >= COPY["jaccard"] for _, v in nbrs[i]))
    comps = components(len(mels), edges)
    res = {"n": n, "with_near_duplicate": has_twin, "with_transposed_copy": has_copy,
           "families": len(comps), "largest_family": max((len(c) for c in comps), default=1)}
    splits = {m.split for m in mels} - {""}
    if splits:
        for s in sorted(splits - {"train"}):
            ids = [i for i in keep if mels[i].split == s]
            leak = [i for i in ids if any(mels[j].split == "train" for j, _ in nbrs[i])]
            copy = [i for i in ids if any(mels[j].split == "train" and v["jaccard"] >= COPY["jaccard"] for j, v in nbrs[i])]
            res[f"official_{s}"] = {"n": len(ids), "twin_in_train": len(leak), "copy_in_train": len(copy),
                                    "examples": [(mels[i].piece_id, mels[i].meta, [(mels[j].piece_id, mels[j].meta, round(v["jaccard"], 3), v["lcs"])
                                                  for j, v in nbrs[i] if mels[j].split == "train"][:2]) for i in copy[:8]]}
    if name == "pdmx":
        ded = [i for i in keep if mels[i].meta.get("dedup")]
        ded_set = set(ded)
        copy_in = [i for i in ded if any(j in ded_set and v["jaccard"] >= COPY["jaccard"] for j, v in nbrs[i])]
        twin_in = [i for i in ded if any(j in ded_set for j, _ in nbrs[i])]
        res["pdmx_dedup_subset"] = {"n": len(ded), "with_transposed_copy_inside_subset": len(copy_in),
                                    "with_near_duplicate_inside_subset": len(twin_in),
                                    "examples": [(mels[i].meta.get("title"), [(mels[j].meta.get("title"), round(v["jaccard"], 3)) for j, v in nbrs[i] if j in ded_set][:2]) for i in copy_in[:10]]}
    # Random 80/10/10 splits, the protocol many papers use for corpora without official splits.
    sims = []
    for _ in range(200):
        role = rng.choice(3, size=len(mels), p=[0.8, 0.1, 0.1])
        test = [i for i in keep if role[i] == 2]
        sims.append(np.mean([any(role[j] == 0 for j, _ in nbrs[i]) for i in test]))
    res["random_split_test_twin_rate"] = {"mean": float(np.mean(sims)), "p5": float(np.percentile(sims, 5)), "p95": float(np.percentile(sims, 95))}
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    corpora = sys.argv[1:] or ["hooktheory", "pop909", "nottingham", "essen"]
    out = {}
    path = ROOT / "results" / "03_corpus_audit.json"
    if path.exists():
        out = json.loads(path.read_text())
    for name in corpora:
        mels = getattr(sources, name)()
        out[name] = audit(name, mels, rng)
        r = out[name]
        print(f"{name:11s} n={r['n']:6d}  any near-dup {r['with_near_duplicate'] / r['n']:.1%}  transposed copy {r['with_transposed_copy'] / r['n']:.1%}  "
              f"families {r['families']} (largest {r['largest_family']})  random-split test twin rate {r['random_split_test_twin_rate']['mean']:.1%}", flush=True)
        for k, v in r.items():
            if k.startswith("official_"):
                print(f"   {k}: n={v['n']} twin in train {v['twin_in_train']} ({v['twin_in_train'] / max(1, v['n']):.1%}), copy in train {v['copy_in_train']} ({v['copy_in_train'] / max(1, v['n']):.1%})")
                for ex in v["examples"][:4]:
                    print("      ", ex)
        path.write_text(json.dumps(out, indent=2, default=str))
