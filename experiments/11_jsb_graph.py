"""Near-duplicate graph statistics for the JSB soprano lines (same detector as the corpus audit),
including the twin rate a random 80/10/10 split would produce, for comparison with the official split."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

if __name__ == "__main__":
    audit = importlib.import_module("03_corpus_audit")
    sop = importlib.import_module("06_jsb_soprano_counterfactual").sopranos()
    r = audit.audit("jsb", [m for v in sop.values() for m in v], np.random.default_rng(0))
    r = {k: v for k, v in r.items() if k != "examples"}
    print(r)
    (ROOT / "results" / "11_jsb_graph.json").write_text(json.dumps(r, indent=1, default=float))
