"""Train a token transformer on melodies with a fixed token budget; evaluate per-note NLL."""
import math
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from . import tokens as T  # noqa: E402
from .models import Transformer  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SIZES = {  # roughly 0.4M, 1.6M, 6.4M, 25M parameters
    "XS": dict(d=96, layers=4, heads=4),
    "S": dict(d=192, layers=4, heads=4),
    "M": dict(d=320, layers=6, heads=8),
    "L": dict(d=512, layers=8, heads=8),
}


def pack(seqs, ctx, rng):
    """Concatenate shuffled token sequences into fixed-length training windows."""
    order = rng.permutation(len(seqs))
    flat = np.concatenate([seqs[i] for i in order]).astype(np.int64)
    n = (len(flat) - 1) // ctx
    x = flat[: n * ctx].reshape(n, ctx)
    y = flat[1: n * ctx + 1].reshape(n, ctx)
    return x, y


def train(train_seqs, size="S", ctx=512, token_budget=30_000_000, bs=32, lr=6e-4, seed=0, log=None, augment=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Transformer("tokens", vocab=T.VOCAB, **SIZES[size], dropout=0.1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps = max(1, token_budget // (bs * ctx))
    warm = max(1, steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    step, t0, epoch = 0, time.time(), 0
    while step < steps:
        seqs = train_seqs(epoch) if callable(train_seqs) else train_seqs
        x, y = pack(seqs, ctx, rng)
        for i in range(0, len(x) - bs + 1, bs):
            xb = torch.tensor(x[i:i + bs], device=DEV)
            yb = torch.tensor(y[i:i + bs], device=DEV)
            loss = F.cross_entropy(model(xb).reshape(-1, T.VOCAB), yb.reshape(-1), ignore_index=T.PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if log and step % 200 == 0:
                log(f"step {step}/{steps} loss {loss.item():.3f} {(time.time() - t0) / step:.3f}s/step")
            if step >= steps:
                break
        epoch += 1
    return model


@torch.no_grad()
def nll_per_note(model, seqs, ctx=512):
    """Mean negative log-likelihood per note (pitch + duration token), in nats. Pieces longer than
    the context are scored in overlapping windows, each token scored once with left context."""
    model.eval()
    total, notes = 0.0, 0
    for s in seqs:
        s = torch.tensor(s.astype(np.int64), device=DEV)
        n = len(s)
        start = 0
        while start < n - 1:
            lo = max(0, start - ctx // 2) if start else 0
            hi = min(n, lo + ctx)
            logits = model(s[lo:hi - 1][None])[0]
            lp = F.log_softmax(logits.float(), -1)
            tgt = s[lo + 1:hi]
            tok_lp = lp.gather(1, tgt[:, None])[:, 0]
            skip = start - lo
            total -= float(tok_lp[skip:].sum())
            start = hi - 1
        notes += (n - 2) / 2
    return total / max(1, notes)


@torch.no_grad()
def generate(model, prompt, n_tokens, temperature=1.0, top_k=0, greedy=False, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    x = torch.tensor(np.asarray(prompt, np.int64), device=DEV)[None]
    for _ in range(n_tokens):
        logits = model(x[:, -512:])[0, -1].float().cpu()
        if greedy:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[-1]] = -float("inf")
            nxt = int(torch.multinomial(F.softmax(logits, -1), 1, generator=g))
        x = torch.cat([x, torch.tensor([[nxt]], device=DEV)], 1)
        if nxt == T.EOS:
            break
    return x[0].cpu().numpy()


@torch.no_grad()
def generate_batch(model, prompts, n_tokens, temperature=1.0, greedy=False, seed=0, chunk=256):
    """Generate continuations for equal-length prompts in batches. Returns a list of arrays."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    P = np.asarray(prompts, np.int64)
    for i in range(0, len(P), chunk):
        x = torch.tensor(P[i:i + chunk], device=DEV)
        for _ in range(n_tokens):
            logits = model(x[:, -512:])[:, -1].float()
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, -1).cpu()
                nxt = torch.multinomial(probs, 1, generator=g).to(DEV)
            x = torch.cat([x, nxt], 1)
        out.extend(x.cpu().numpy())
    return out
