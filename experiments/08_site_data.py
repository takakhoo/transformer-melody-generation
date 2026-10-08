"""Compact JSON for the interactive pages: JSB test chorales with their training twins (sopranos as
notes), PDMX copies that survive PDMX's own deduplication, the audit table, and the counterfactual
results. Essen melodies are never exported (its licence forbids redistribution)."""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "site" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def notes(m, limit=64):
    p, o, d = m.notes()
    o = o - (o[0] if len(o) else 0)
    return [[int(a), round(float(b), 3), round(float(c), 3)] for a, b, c in zip(p[:limit], o[:limit], d[:limit])]


def jsb_pairs():
    import importlib
    sop = importlib.import_module("06_jsb_soprano_counterfactual").sopranos()
    audit = json.loads((ROOT / "results" / "02_jsb_identify.json").read_text())
    pairs = []
    for it in audit["test"]["items"]:
        if it["jaccard"] < 0.5:
            continue
        ti = int(it["id"].split("/")[1])
        tr = int(it["train_twin"].split("/")[1])
        a, b = sop["test"][ti], sop["train"][tr]
        shift = int(np.median(a.notes()[0]) - np.median(b.notes()[0]))
        pairs.append({"test": it["id"], "train": it["train_twin"], "bwv_test": it["bwv"], "bwv_train": it["twin_bwv"],
                      "jaccard": it["jaccard"], "run": it["lcs_frac"], "transpose": shift,
                      "a": notes(a), "b": notes(b)})
    return sorted(pairs, key=lambda p: -p["jaccard"])


def pdmx_copies(k=12):
    fam = pickle.loads((ROOT / "results" / "memorization" / "families.pkl").read_bytes())
    mels = fam["mels"]
    audit = json.loads((ROOT / "results" / "03_corpus_audit.json").read_text())["pdmx"]["pdmx_dedup_subset"]["examples"]
    by_title = {}
    for m in mels:
        if m.meta.get("dedup"):
            by_title.setdefault(m.meta.get("title"), m)
    out = []
    for title, twins in audit:
        if not twins:
            continue
        t2, j = twins[0]
        a, b = by_title.get(title), by_title.get(t2)
        if a is None or b is None or j < 0.9:
            continue
        out.append({"title_a": title, "title_b": t2, "jaccard": j, "a": notes(a, 48), "b": notes(b, 48),
                    "transpose": int(np.median(a.notes()[0]) - np.median(b.notes()[0]))})
        if len(out) >= k:
            break
    return out


if __name__ == "__main__":
    a = json.loads((ROOT / "results" / "03_corpus_audit.json").read_text())
    j = json.loads((ROOT / "results" / "02_jsb_identify.json").read_text())
    bench = json.loads((ROOT / "results" / "01_benchmark_leakage_first_look.json").read_text())
    table = [
        {"corpus": "JSB Chorales", "split": "official test", "n": j["test"]["n"], "twin": j["test"]["jaccard>=0.5"] / j["test"]["n"],
         "copy": j["test"]["jaccard>=0.9"] / j["test"]["n"], "note": "soprano line; 43% share half their soprano"},
        {"corpus": "Essen folksongs", "split": "random 80/10/10", "n": a["essen"]["n"], "twin": a["essen"]["random_split_test_twin_rate"]["mean"],
         "copy": a["essen"]["with_transposed_copy"] / a["essen"]["n"], "note": "share of all songs with a transposed copy"},
        {"corpus": "Nottingham", "split": "random 80/10/10", "n": a["nottingham"]["n"], "twin": a["nottingham"]["random_split_test_twin_rate"]["mean"],
         "copy": a["nottingham"]["with_transposed_copy"] / a["nottingham"]["n"], "note": "official Boulanger split: 2.4%"},
        {"corpus": "PDMX", "split": "random 80/10/10", "n": a["pdmx"]["n"], "twin": a["pdmx"]["random_split_test_twin_rate"]["mean"],
         "copy": a["pdmx"]["with_transposed_copy"] / a["pdmx"]["n"], "note": "6.2% inside PDMX's own deduplicated subset"},
        {"corpus": "Hooktheory", "split": "official test", "n": a["hooktheory"]["official_test"]["n"],
         "twin": a["hooktheory"]["official_test"]["twin_in_train"] / a["hooktheory"]["official_test"]["n"],
         "copy": a["hooktheory"]["official_test"]["copy_in_train"] / a["hooktheory"]["official_test"]["n"], "note": "song-level split; leaks are covers and misspellings"},
        {"corpus": "POP909", "split": "random 80/10/10", "n": a["pop909"]["n"], "twin": a["pop909"]["random_split_test_twin_rate"]["mean"],
         "copy": a["pop909"]["with_transposed_copy"] / a["pop909"]["n"], "note": ""},
    ]
    cf = {}
    for name in ("06_jsb_soprano_counterfactual.json", "07_jsb_soprano_counterfactual.json", "04_jsb_counterfactual.json"):
        p = ROOT / "results" / name
        if p.exists():
            cf[name] = json.loads(p.read_text())
    json.dump({"table": table, "jsb_pairs": jsb_pairs(), "pdmx_copies": pdmx_copies()}, open(OUT / "twins.json", "w"), separators=(",", ":"))
    json.dump({"counterfactual": cf}, open(OUT / "counterfactual.json", "w"), separators=(",", ":"), default=float)
    for f in OUT.iterdir():
        print(f.name, f.stat().st_size // 1024, "KB")
