"""The JSB leak lives in the soprano. Same difference-in-differences design as experiment 04, on the
soprano line alone: a token language model over (pitch, duration) notes of the top voice, trained
with and without the training chorales whose soprano twins a test chorale. Also an n-gram baseline
and a nearest-neighbour copy model, which can only win by retrieving a twin."""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melody import tokens as T  # noqa: E402
from melody.representation import Melody  # noqa: E402


def sopranos():
    m = sio.loadmat(ROOT / "data" / "raw" / "boulanger2012" / "tcn_JSB_Chorales.mat")
    out = {}
    for split, key in (("train", "traindata"), ("valid", "validdata"), ("test", "testdata")):
        rows = []
        for roll in m[key][0]:
            roll = np.asarray(roll, bool)
            top = np.where(roll.any(1), 87 - np.argmax(roll[:, ::-1], 1) + 21, -1)
            # Frames to notes: a change of top pitch starts a note (repeated pitches merge, as the roll cannot tell them apart).
            pitch, onset, dur = [], [], []
            for t, p in enumerate(top):
                if pitch and p == pitch[-1]:
                    dur[-1] += 1
                else:
                    pitch.append(int(p)); onset.append(float(t)); dur.append(1.0)
            rows.append(Melody(np.array(pitch), np.array(onset), np.array(dur), "jsb"))
        out[split] = rows
    return out


class NGram:
    """Interpolated Kneser-Ney-style backoff over note tokens (pitch*100+duration index)."""
    def __init__(self, order=4, d=0.75):
        self.order, self.d = order, d
        self.counts = [defaultdict(Counter) for _ in range(order)]
        self.vocab = set()

    def fit(self, seqs):
        for s in seqs:
            s = ["<s>"] * (self.order - 1) + list(s) + ["</s>"]
            for i in range(self.order - 1, len(s)):
                self.vocab.add(s[i])
                for k in range(self.order):
                    self.counts[k][tuple(s[i - k:i])][s[i]] += 1
        return self

    def prob(self, ctx, w):
        p = 1.0 / (len(self.vocab) + 1)
        for k in range(self.order):
            c = self.counts[k].get(tuple(ctx[len(ctx) - k:]) if k else (), None)
            if not c:
                continue
            tot = sum(c.values())
            p = max(c.get(w, 0) - self.d, 0) / tot + self.d * len(c) / tot * p
        return p

    def nll(self, s):
        s = ["<s>"] * (self.order - 1) + list(s) + ["</s>"]
        return -sum(np.log(self.prob(s[i - self.order + 1:i], s[i])) for i in range(self.order - 1, len(s))) / (len(s) - self.order + 1)


def note_symbols(m):
    toks = T.encode(m)[1:-1]
    return [int(a) * 100 + int(b) for a, b in zip(toks[::2], toks[1::2])]


if __name__ == "__main__":
    from melody import train_lm
    data = sopranos()
    audit = json.loads((ROOT / "results" / "02_jsb_identify.json").read_text())
    leaked = {int(i["id"].split("/")[1]) for i in audit["test"]["items"] if i["jaccard"] >= 0.5}
    twins = {int(i["train_twin"].split("/")[1]) for i in audit["test"]["items"] if i["jaccard"] >= 0.5}
    is_leak = np.array([i in leaked for i in range(len(data["test"]))])
    arms = {"A_full": data["train"], "B_no_twins": [m for i, m in enumerate(data["train"]) if i not in twins]}
    rows = []
    for arm, train in arms.items():
        ng = NGram(4).fit([note_symbols(m) for m in train])
        nll = np.array([ng.nll(note_symbols(m)) for m in data["test"]])
        rows.append({"model": "4-gram", "arm": arm, "seed": 0, "leaked": nll[is_leak].mean(), "clean": nll[~is_leak].mean()})
        for size in ("XS", "S"):
            for seed in range(3):
                seqs = [T.encode(m) for m in train]

                def epochs(e, train=train, seed=seed):
                    r = np.random.default_rng(seed * 100 + e)
                    return [T.encode(m, transpose=int(r.integers(-5, 7))) for m in train]
                model = train_lm.train(epochs, size=size, ctx=256, token_budget=1_500_000, bs=16, lr=5e-4, seed=seed)
                per = np.array([train_lm.nll_per_note(model, [T.encode(m)]) for m in data["test"]])
                rows.append({"model": f"transformer-{size}", "arm": arm, "seed": seed, "leaked": per[is_leak].mean(), "clean": per[~is_leak].mean()})
                print(rows[-1], flush=True)
    df = {}
    for r in rows:
        df.setdefault((r["model"], r["seed"]), {})[r["arm"]] = r
    out = []
    for (model, seed), d in df.items():
        if len(d) == 2:
            did = (d["B_no_twins"]["leaked"] - d["A_full"]["leaked"]) - (d["B_no_twins"]["clean"] - d["A_full"]["clean"])
            out.append({"model": model, "seed": seed, "A_leaked": d["A_full"]["leaked"], "A_clean": d["A_full"]["clean"],
                        "B_leaked": d["B_no_twins"]["leaked"], "B_clean": d["B_no_twins"]["clean"], "did_nats_per_note": did})
            print(f"{model:15s} seed{seed}: full-train leaked {d['A_full']['leaked']:.3f} clean {d['A_full']['clean']:.3f} | no-twins leaked {d['B_no_twins']['leaked']:.3f} clean {d['B_no_twins']['clean']:.3f} | DiD {did:+.3f}")
    (ROOT / "results" / "06_jsb_soprano_counterfactual.json").write_text(json.dumps(out, indent=2, default=float))
