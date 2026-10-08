"""Does split leakage inflate JSB Chorales likelihoods? Difference-in-differences with retraining.

Arm A trains on the standard training split. Arm B trains on the same split minus every
training chorale whose soprano is a transposed near-duplicate (J >= 0.5) of some test chorale.
Both are evaluated on the standard test split, separated into leaked pieces (those with a twin)
and clean pieces. Inflation on leaked pieces, net of the change on clean pieces, measures what
the leak is worth. Repeated over model sizes and seeds."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melody.models import Transformer, frame_nll  # noqa: E402

DEV = "cpu"  # faster than MPS for these short, many-step runs
torch.set_num_threads(4)
WIN = 128
SIZES = {"S": dict(d=64, layers=2, heads=2), "M": dict(d=160, layers=4, heads=4), "L": dict(d=320, layers=6, heads=8)}


def load():
    m = sio.loadmat(ROOT / "data" / "raw" / "boulanger2012" / "tcn_JSB_Chorales.mat")
    return {s: [np.asarray(r, np.float32) for r in m[k][0]] for s, k in (("train", "traindata"), ("valid", "validdata"), ("test", "testdata"))}


def batchify(rolls, aug=False, rng=None):
    T = max(len(r) for r in rolls)
    x = np.zeros((len(rolls), T, 88), np.float32)
    mask = np.zeros((len(rolls), T), np.float32)
    for i, r in enumerate(rolls):
        if aug:
            s = int(rng.integers(-5, 7))
            r = np.roll(r, s, axis=1)
            if s > 0:
                r[:, :s] = 0
            elif s < 0:
                r[:, s:] = 0
        x[i, :len(r)] = r
        mask[i, :len(r)] = 1
    return torch.tensor(x, device=DEV), torch.tensor(mask, device=DEV)


def piece_nll(model, rolls):
    """Mean NLL per frame, scored in windows of WIN frames with a stride of WIN/2 so every frame
    after the first window sees at least WIN/2 frames of context."""
    model.eval()
    out = []
    with torch.no_grad():
        for r in rolls:
            n, total, start = len(r), 0.0, 0
            while start < n:
                lo = 0 if start == 0 else start - WIN // 2
                hi = min(n, lo + WIN)
                x, m = batchify([r[lo:hi]])
                _, nll = frame_nll(model, x, m)
                total += float(nll[0, start - lo: hi - lo].sum())
                start = hi
            out.append(total / n)
    return np.array(out)


def train(rolls, valid, size, seed, steps=3000, lr=1e-3, bs=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Transformer("frames", **SIZES[size], dropout=0.2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, best_state, wait = np.inf, None, 0
    for step in range(steps):
        model.train()
        crops = []
        for j in rng.integers(len(rolls), size=bs):
            r = rolls[j]
            s = int(rng.integers(0, max(1, len(r) - WIN + 1)))
            crops.append(r[s:s + WIN])
        x, m = batchify(crops, aug=True, rng=rng)
        loss, _ = frame_nll(model, x, m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 150 == 149:
            v = piece_nll(model, valid).mean()
            if v < best - 1e-3:
                best, best_state, wait = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 5:
                    break
    model.load_state_dict(best_state)
    return model, best


if __name__ == "__main__":
    data = load()
    audit = json.loads((ROOT / "results" / "02_jsb_identify.json").read_text())
    leaked_test = {int(i["id"].split("/")[1]) for i in audit["test"]["items"] if i["jaccard"] >= 0.5}
    twins = {int(i["train_twin"].split("/")[1]) for i in audit["test"]["items"] if i["jaccard"] >= 0.5}
    # Valid pieces whose twin is in train would bias early stopping toward memorization in both arms equally;
    # keep the standard validation split unchanged so the only difference between arms is the removed twins.
    test = data["test"]
    is_leak = np.array([i in leaked_test for i in range(len(test))])
    train_a = data["train"]
    train_b = [r for i, r in enumerate(data["train"]) if i not in twins]
    print(f"test: {is_leak.sum()} leaked / {len(test)}; removing {len(twins)} training twins ({len(train_a)} -> {len(train_b)})", flush=True)
    out = ROOT / "results" / "04_jsb_counterfactual.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["size"], r["seed"]) for r in rows}
    for size in sys.argv[1:] or SIZES:
        for seed in range(3):
            if (size, seed) in done:
                continue
            res = {}
            for arm, tr in (("A_full", train_a), ("B_no_twins", train_b)):
                model, v = train(tr, data["valid"], size, seed)
                nll = piece_nll(model, test)
                res[arm] = {"valid": float(v), "test_all": float(nll.mean()), "test_leaked": float(nll[is_leak].mean()),
                            "test_clean": float(nll[~is_leak].mean()), "params": model.n_params()}
            did = (res["B_no_twins"]["test_leaked"] - res["A_full"]["test_leaked"]) - (res["B_no_twins"]["test_clean"] - res["A_full"]["test_clean"])
            row = {"size": size, "seed": seed, **{f"{a}_{k}": v for a, d in res.items() for k, v in d.items()}, "did_nats_per_frame": did}
            rows.append(row)
            print(f"{size} seed{seed} params={res['A_full']['params']:,}  A: all {res['A_full']['test_all']:.3f} leaked {res['A_full']['test_leaked']:.3f} clean {res['A_full']['test_clean']:.3f}"
                  f" | B: leaked {res['B_no_twins']['test_leaked']:.3f} clean {res['B_no_twins']['test_clean']:.3f} | DiD {did:+.3f}", flush=True)
            out.write_text(json.dumps(rows, indent=2))
