"""Recall of the MinHash LSH candidate stage against exhaustive pairwise comparison, on the two
corpora small enough to compare every pair (JSB sopranos, Nottingham)."""
import importlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from melody import dedup, sources  # noqa: E402
from melody.representation import interval_rhythm_tokens  # noqa: E402

if __name__ == "__main__":
    audit = importlib.import_module("03_corpus_audit")
    jsb = importlib.import_module("06_jsb_soprano_counterfactual").sopranos()
    res = {}
    for name, mels in (("jsb_soprano", [m for s in jsb.values() for m in s]), ("nottingham", sources.nottingham())):
        keep, edges = audit.graph(mels)
        sh = {i: dedup.shingles(interval_rhythm_tokens(mels[i]), audit.K) for i in keep}
        true = {(a, b) for a, b in itertools.combinations(keep, 2) if dedup.jaccard(sh[a], sh[b]) >= 0.5}
        hi = {e for e in true if dedup.jaccard(sh[e[0]], sh[e[1]]) >= 0.9}
        found = {e for e, v in edges.items() if v["jaccard"] >= 0.5}
        res[name] = {"n": len(keep), "pairs_J>=0.5": len(true), "recall_J>=0.5": len(found & true) / max(1, len(true)),
                     "recall_J>=0.9": len(found & hi) / max(1, len(hi))}
        print(name, res[name])
    (ROOT / "results" / "10_lsh_recall.json").write_text(json.dumps(res, indent=1))
