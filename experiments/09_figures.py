"""Paper figures for the leakage audit and the counterfactuals."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})
BLUE, ORANGE, GREY, RED = "#1f5fa8", "#d68910", "#8a8a8a", "#c0392b"


def save(fig, name):
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png")
    plt.close(fig)


def audit():
    t = json.loads((ROOT / "site" / "data" / "twins.json").read_text())["table"]
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    names = [f"{r['corpus']}\n({r['split']})" for r in t]
    y = np.arange(len(t))[::-1]
    ax.barh(y, [r["twin"] * 100 for r in t], color=BLUE, height=0.55, label="test melodies with a near-duplicate in training")
    ax.barh(y, [r["copy"] * 100 for r in t], color=ORANGE, height=0.55, label="transposed copy")
    for yy, r in zip(y, t):
        ax.text(r["twin"] * 100 + 0.8, yy, f"{r['twin'] * 100:.1f}%", va="center", fontsize=7)
    ax.set_yticks(y, names, fontsize=7)
    ax.set_xlabel("Percent")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig1_audit")


def jsb_pair():
    t = json.loads((ROOT / "site" / "data" / "twins.json").read_text())["jsb_pairs"]
    p = next(x for x in t if x["test"] == "test/43")
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 2.4), sharex=True)
    ps = [n[0] for n in p["a"] + p["b"] if n[1] <= 32]
    lo, hi = min(ps) - 1, max(ps) + 1
    names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    for ax, notes, col, lab in ((axes[0], p["a"], BLUE, f"Test chorale {p['test'].split('/')[1]}"),
                                (axes[1], p["b"], ORANGE, f"Training chorale {p['train'].split('/')[1]}, a different harmonization of the same hymn tune")):
        for pi, on, du in notes:
            if on > 32:
                break
            ax.add_patch(plt.Rectangle((on, pi - 0.4), du * 0.95, 0.8, color=col))
        ax.set_ylim(lo, hi)
        ticks = [t for t in range(lo, hi + 1) if t % 12 in (0, 4, 7)]
        ax.set_yticks(ticks, [f"{names[t % 12]}{t // 12 - 1}" for t in ticks], fontsize=6.5)
        ax.set_title(lab, fontsize=7, loc="left", pad=2)
    axes[1].set_xlim(0, 32)
    axes[1].set_xlabel("Beats (one frame per quarter note, as in the benchmark)")
    save(fig, "fig2_jsb_pair")


def counterfactual():
    d = json.loads((R / "07_jsb_soprano_counterfactual.json").read_text())
    J = np.array(d["jaccard"])
    leak = J >= 0.5
    from collections import defaultdict
    acc = defaultdict(list)
    for r in d["rows"]:
        acc[(r["model"], r["arm"])].append(np.array(r["per_piece"]))
    models = ["4-gram", "transformer-XS", "transformer-S", "transformer-M"]
    labels = ["4-gram", "TF 0.46M", "TF 1.8M", "TF 7.4M"]
    did, err, a_all, a_clean, b_all = [], [], [], [], []
    for m in models:
        A, B = np.array(acc[(m, "A_full")]), np.array(acc[(m, "B_no_twins")])
        per_seed = [((b - a)[leak].mean() - (b - a)[~leak].mean()) for a, b in zip(A, B)]
        did.append(np.mean(per_seed))
        err.append(np.std(per_seed) if len(per_seed) > 1 else 0)
        a_all.append(A.mean(0).mean())
        a_clean.append(A.mean(0)[~leak].mean())
        b_all.append(B.mean(0).mean())
    fig, (x1, x2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x1.bar(range(4), did, yerr=err, color=[GREY, BLUE, BLUE, BLUE], capsize=2, width=0.6)
    x1.set_yscale("log")
    x1.set_xticks(range(4), labels, fontsize=7)
    x1.set_ylabel("Leak benefit (nats/note, log)")
    for i, v in enumerate(did):
        x1.text(i, v * 1.15, f"{v:.3f}", ha="center", fontsize=6.5)
    w = 0.27
    x2.bar(np.arange(4) - w, a_all, w, color=RED, label="standard test set (leaky)")
    x2.bar(np.arange(4), a_clean, w, color=BLUE, label="clean test chorales only")
    x2.bar(np.arange(4) + w, b_all, w, color=GREY, label="trained without the twins")
    x2.set_ylim(2.0, 3.2)
    x2.set_xticks(range(4), labels, fontsize=7)
    x2.set_ylabel("Test NLL (nats/note)")
    x2.legend(fontsize=6.3, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig3_counterfactual")


if __name__ == "__main__":
    for fn in (audit, jsb_pair, counterfactual):
        fn()
        print("ok", fn.__name__)
