"""Release files: twin lists per corpus and family-level splits in which no test melody has a twin
in training. Essen is listed by identifier only (its terms forbid redistributing the melodies)."""
import csv
import gzip
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from melody import sources  # noqa: E402

OUT = ROOT / "release"
audit = importlib.import_module("03_corpus_audit")


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    return np.array([find(i) for i in range(n)])


def family_split(fam, rng, frac=(0.8, 0.1, 0.1)):
    """Assign whole families to train/valid/test, filling test and valid first to their melody quotas."""
    uniq, counts = np.unique(fam, return_counts=True)
    order = rng.permutation(len(uniq))
    n = len(fam)
    quota = {"test": frac[2] * n, "valid": frac[1] * n}
    split_of, filled = {}, {"test": 0, "valid": 0}
    for k in order:
        for s in ("test", "valid"):
            if filled[s] + counts[k] <= quota[s] * 1.02 and filled[s] < quota[s]:
                split_of[uniq[k]] = s
                filled[s] += counts[k]
                break
        else:
            split_of[uniq[k]] = "train"
    return [split_of[f] for f in fam]


def release(name, mels, ids, write_split=True, seed=0):
    keep, edges = audit.graph(mels)
    with open(OUT / "twins" / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id_a", "id_b", "jaccard", "longest_common_run", "transposition_semitones"])
        for (a, b), v in sorted(edges.items(), key=lambda e: -e[1]["jaccard"]):
            pa, pb = mels[a].notes()[0], mels[b].notes()[0]
            shift = int(np.median(pa) - np.median(pb)) if len(pa) and len(pb) else 0
            w.writerow([ids[a], ids[b], round(v["jaccard"], 3), v["lcs"], shift])
    res = {"melodies": len(mels), "twin_pairs": len(edges)}
    if write_split:
        fam = components(len(mels), edges)
        split = family_split(fam, np.random.default_rng(seed))
        (OUT / "splits" / f"{name}_family_split.json").write_text(json.dumps(dict(zip(ids, split))))
        tr = {i for i, s in enumerate(split) if s == "train"}
        leak = sum(1 for (a, b) in edges if (a in tr) != (b in tr) and "test" in (split[a], split[b]))
        res.update({s: split.count(s) for s in ("train", "valid", "test")}, cross_split_test_twins=leak)
    print(name, res, flush=True)
    return res


if __name__ == "__main__":
    (OUT / "twins").mkdir(parents=True, exist_ok=True)
    (OUT / "splits").mkdir(parents=True, exist_ok=True)
    summary = {}
    sop = importlib.import_module("06_jsb_soprano_counterfactual").sopranos()
    jsb, jsb_ids = [], []
    for split, ms in sop.items():
        for i, m in enumerate(ms):
            jsb.append(m)
            jsb_ids.append(f"{split}/{i}")
    summary["jsb"] = release("jsb", jsb, jsb_ids)
    j = json.loads((ROOT / "results" / "02_jsb_identify.json").read_text())
    clean = [it["id"] for it in j["test"]["items"] if it["jaccard"] < 0.5]
    (OUT / "splits" / "jsb_clean_test.txt").write_text("\n".join(sorted(clean, key=lambda s: int(s.split("/")[1]))) + "\n")
    summary["jsb_clean_test"] = len(clean)
    for name, fn, split in (("nottingham", sources.nottingham, True), ("hooktheory", sources.hooktheory, False),
                            ("pop909", sources.pop909, False), ("essen", sources.essen, False), ("pdmx", sources.pdmx, True)):
        ms = fn()
        summary[name] = release(name, ms, [m.piece_id for m in ms], write_split=split)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for f in list((OUT / "twins").iterdir()) + list((OUT / "splits").iterdir()):
        if f.suffix != ".gz" and f.stat().st_size > 5_000_000:  # keep the repo small: gzip the PDMX files
            with open(f, "rb") as src, gzip.open(f"{f}.gz", "wb") as dst:
                dst.write(src.read())
            f.unlink()
