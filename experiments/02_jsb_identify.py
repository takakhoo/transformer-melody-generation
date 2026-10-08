"""Identify every JSB Chorales benchmark piece by BWV number and chorale tune, by matching its
transposition-invariant soprano line against the music21 Bach chorale corpus, then list the test
pieces whose tune also appears in the training split. Exhaustive pairwise comparison, no LSH."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melody import dedup  # noqa: E402
from melody.representation import frame_tokens  # noqa: E402

from music21 import corpus  # noqa: E402


def benchmark():
    m = sio.loadmat(ROOT / "data" / "raw" / "boulanger2012" / "tcn_JSB_Chorales.mat")
    out = []
    for split, key in (("train", "traindata"), ("valid", "validdata"), ("test", "testdata")):
        for i, roll in enumerate(m[key][0]):
            roll = np.asarray(roll, bool)
            top = np.where(roll.any(1), 87 - np.argmax(roll[:, ::-1], 1), -1).astype(float)
            top[top < 0] = np.nan
            out.append({"id": f"{split}/{i}", "split": split, "soprano": top})
    return out


def corpus_sopranos():
    """Soprano (part 0) at one frame per quarter note, the benchmark's resolution."""
    rows = []
    for path in corpus.getComposer("bach"):
        try:
            s = corpus.parse(path)
        except Exception:
            continue
        part = s.parts[0]
        frames = np.full(int(np.ceil(float(part.highestTime))), np.nan)
        for n in part.flatten().notes:
            p = max(x.midi for x in n.pitches) if n.isChord else n.pitch.midi
            a, b = int(round(float(n.offset))), int(round(float(n.offset + n.quarterLength)))
            frames[a:max(a + 1, b)] = p
        rows.append({"bwv": Path(str(path)).stem, "title": "", "soprano": frames})
    return rows


if __name__ == "__main__":
    bench = benchmark()
    ref = corpus_sopranos()
    k = 8
    for r in bench + ref:
        r["tok"] = frame_tokens(r["soprano"])
        r["sh"] = dedup.shingles(r["tok"], k)
    # 1) Identify each benchmark piece with its best corpus match.
    for b in bench:
        scores = [(dedup.jaccard(b["sh"], c["sh"]), c) for c in ref]
        j, c = max(scores, key=lambda x: x[0])
        b["bwv"], b["title"], b["match_j"] = c["bwv"], c["title"], j
    # 2) Exhaustive cross-split comparison of benchmark sopranos.
    train = [b for b in bench if b["split"] == "train"]
    report = {}
    for split in ("valid", "test"):
        items = []
        for b in (x for x in bench if x["split"] == split):
            best = max(train, key=lambda t: dedup.jaccard(b["sh"], t["sh"]))
            v = dedup.verify(b["tok"], best["tok"], b["sh"], best["sh"])
            items.append({"id": b["id"], "bwv": b["bwv"], "title": b["title"], "match_j": round(b["match_j"], 3),
                          "train_twin": best["id"], "twin_bwv": best["bwv"], "twin_title": best["title"],
                          "jaccard": round(v["jaccard"], 3), "lcs_frac": round(v["lcs_frac"], 3)})
        n = len(items)
        report[split] = {
            "n": n,
            "jaccard>=0.9": sum(i["jaccard"] >= 0.9 for i in items),
            "jaccard>=0.5": sum(i["jaccard"] >= 0.5 for i in items),
            "lcs_frac>=0.5": sum(i["lcs_frac"] >= 0.5 for i in items),
            "same_title_in_train": sum(bool(i["title"]) and i["title"] == i["twin_title"] for i in items),
            "items": sorted(items, key=lambda i: -i["jaccard"]),
        }
    ident = np.array([b["match_j"] for b in bench])
    report["identification"] = {"n": len(bench), "match_j>=0.9": int((ident >= 0.9).sum()), "median_match_j": float(np.median(ident))}
    (ROOT / "results" / "02_jsb_identify.json").write_text(json.dumps(report, indent=2))
    print("identification:", report["identification"])
    for split in ("valid", "test"):
        r = report[split]
        print(f"{split}: n={r['n']}  J>=0.9 {r['jaccard>=0.9']} ({r['jaccard>=0.9'] / r['n']:.1%})  J>=0.5 {r['jaccard>=0.5']} ({r['jaccard>=0.5'] / r['n']:.1%})"
              f"  run>=50% {r['lcs_frac>=0.5']} ({r['lcs_frac>=0.5'] / r['n']:.1%})")
        for i in r["items"][:12]:
            print(f"   {i['id']:9s} {i['bwv']:12s} '{i['title'][:38]}'  <- train {i['train_twin']:10s} {i['twin_bwv']:12s} '{i['twin_title'][:38]}'  J={i['jaccard']} run={i['lcs_frac']}")
