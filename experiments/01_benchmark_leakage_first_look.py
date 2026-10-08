"""First look: transposed and near-duplicate pieces across the standard train/valid/test splits of
the four Boulanger-Lewandowski et al. (2012) piano-roll benchmarks."""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melody import dedup  # noqa: E402
from melody.representation import frame_tokens  # noqa: E402

RAW = ROOT / "data" / "raw" / "boulanger2012"
FILES = {"Nottingham": "tcn_Nottingham.mat", "JSB Chorales": "tcn_JSB_Chorales.mat",
         "MuseData": "tcn_MuseData.mat", "Piano-midi.de": "tcn_Piano_midi.mat"}


def load(name):
    m = sio.loadmat(RAW / FILES[name])
    out = []
    for split, key in (("train", "traindata"), ("valid", "validdata"), ("test", "testdata")):
        for i, roll in enumerate(m[key][0]):
            out.append((f"{split}/{i}", split, np.asarray(roll, bool)))
    return out


def top_line(roll):
    idx = np.where(roll.any(1), 87 - np.argmax(roll[:, ::-1], 1), -1).astype(float)
    idx[idx < 0] = np.nan
    return idx


def poly_tokens(roll):
    """Transposition-invariant frame symbols: bass motion plus the interval set above the bass."""
    toks, prev_bass = [], None
    for frame in roll:
        on = np.flatnonzero(frame)
        if len(on) == 0:
            toks.append(-7)
            prev_bass = None
            continue
        bass = on[0]
        motion = 0 if prev_bass is None else int(np.clip(bass - prev_bass, -24, 24))
        chord = sum(1 << int(min(i, 30)) for i in (on[1:] - bass))
        toks.append(hash((motion, chord)) % (1 << 40))
        prev_bass = bass
    return np.array(toks, dtype=np.int64)


def audit(name, k_top=16, k_poly=8):
    pieces = load(name)
    lsh = dedup.MinHashLSH(n_perm=128, bands=32)
    rows = {}
    for view, fn, k in (("top_line", lambda r: frame_tokens(top_line(r)), k_top), ("polyphonic", poly_tokens, k_poly)):
        seqs = {pid: fn(roll) for pid, _, roll in pieces}
        sh = {pid: dedup.shingles(s, k) for pid, s in seqs.items()}
        sigs = {pid: lsh.signature(s) for pid, s in sh.items()}
        cand = lsh.candidates(sigs)
        split = {pid: sp for pid, sp, _ in pieces}
        best = {}
        for a, b in cand:
            if split[a] == split[b]:
                continue
            v = dedup.verify(seqs[a], seqs[b], sh[a], sh[b])
            for x, y in ((a, b), (b, a)):
                if split[x] != "train" and split[y] == "train":
                    if x not in best or v["jaccard"] > best[x][1]["jaccard"]:
                        best[x] = (y, v)
        n_eval = {s: sum(1 for _, sp, _ in pieces if sp == s) for s in ("valid", "test")}
        res = {}
        for s in ("valid", "test"):
            ids = [pid for pid, (_, v) in best.items() if split[pid] == s]
            jac = np.array([best[p][1]["jaccard"] for p in ids])
            frac = np.array([best[p][1]["lcs_frac"] for p in ids])
            res[s] = {"n": n_eval[s],
                      "jaccard>=0.9": int((jac >= 0.9).sum()), "jaccard>=0.5": int((jac >= 0.5).sum()),
                      "jaccard>=0.3": int((jac >= 0.3).sum()), "lcs_frac>=0.5": int((frac >= 0.5).sum())}
        rows[view] = {"splits": res, "examples": sorted(((p, best[p][0], round(best[p][1]["jaccard"], 3), round(best[p][1]["lcs_frac"], 3))
                                                         for p in best if split[p] == "test"), key=lambda r: -r[2])[:15]}
    return rows


if __name__ == "__main__":
    out = {name: audit(name) for name in FILES}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "01_benchmark_leakage_first_look.json").write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        for view, v in r.items():
            t = v["splits"]["test"]
            print(f"{name:14s} {view:10s} test n={t['n']:4d}  J>=0.9: {t['jaccard>=0.9']:4d} ({t['jaccard>=0.9'] / t['n']:.1%})  "
                  f"J>=0.5: {t['jaccard>=0.5']:4d} ({t['jaccard>=0.5'] / t['n']:.1%})  J>=0.3: {t['jaccard>=0.3'] / t['n']:.1%}  run>=50%: {t['lcs_frac>=0.5'] / t['n']:.1%}")
        print("   top examples (test, train, jaccard, run):", r["top_line"]["examples"][:4])
