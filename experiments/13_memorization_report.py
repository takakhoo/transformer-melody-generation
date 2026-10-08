"""Collect the memorization runs into one table, a LaTeX table for the paper, and a figure:
canary exposure (NLL of never-inserted canaries minus NLL of inserted ones) against the number of
insertions, and extraction rates."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "memorization"
DUPS = (1, 2, 4, 8, 16, 32)
PARAMS = {"XS": "0.46M", "S": "1.8M", "M": "7.4M", "L": "25M"}


def label(r):
    s = f"{PARAMS[r['size']]} {r['corpus']}"
    if not r.get("augment", True):
        s += ", no aug."
    if r.get("train_frac", 0.15) != 0.15:
        s += f", {r['train_melodies'] / 1000:.1f}k mel."
    return s


def rows():
    cp = ROOT / "results" / "14_copying.json"
    copying = json.loads(cp.read_text()) if cp.exists() else {}
    out = []
    for p in sorted(M.glob("*.json")):
        r = json.loads(p.read_text())
        r["tag"] = p.stem
        r["augment"] = r.get("augment", True)
        r["train_frac"] = r.get("train_frac", 0.15)
        r["exposure"] = {d: r["canary_nll_x0"] - r[f"canary_nll_x{d}"] for d in DUPS}
        r["max_extract"] = max(r[f"canary_extract_x{d}"] for d in DUPS)
        r["copying"] = copying.get(p.stem, {}).get("0.8")
        out.append(r)
    order = {"XS": 0, "S": 1, "M": 2, "L": 3}
    return sorted(out, key=lambda r: (not r["augment"], r["train_frac"] != 0.15, order[r["size"]], r["corpus"]))


def latex(rs):
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"\colhead{Run} & \colhead{Test NLL} & \colhead{Exposure $\times$1} & \colhead{Exposure $\times$32}"
             r" & \colhead{Extract $\times$32} & \colhead{Natural extract} & \colhead{Any run $\geq$20} & \colhead{Longest non-trivial} \\", r"\midrule"]
    for r in rs:
        nat = max(r[k] for k in r if k.startswith("natural_extract_"))
        c = r["copying"]
        anyrun = f"{100 * c['strict_run>=20']:.0f}\\%" if c else "--"
        longest = f"{c['max_nontrivial']}" if c else "--"
        lines.append(f"{label(r)} & {r['test_nll_per_note']:.3f} & {r['exposure'][1]:+.3f} & "
                     f"{r['exposure'][32]:+.3f} & {100 * r['canary_extract_x32']:.0f}\\% & {100 * nat:.1f}\\% & {anyrun} & {longest} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Memorization on PDMX. \textit{Exposure}: NLL of 60 never-inserted canaries minus NLL of 60 canaries inserted 1, 8 or 32 times, in nats per note (positive means the model prefers the inserted canaries). \textit{Extract}: share of canaries inserted 32 times whose next 20 notes greedy decoding reproduces exactly from a 12-note prompt. \textit{Natural extract}: the highest extraction rate over real training melodies grouped by family size. \textit{Any run}: share of 300 unconditional samples at temperature 0.8 that share a contiguous passage of at least 20 notes with one training melody, up to transposition. \textit{Longest non-trivial}: the longest such passage, in notes, among passages with at least four distinct intervals and at most half repeated notes. Test NLL is on held-out families and is comparable only between runs with the same training pool.}",
              r"\label{tab:mem}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def figure(rs):
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    cmap = plt.get_cmap("viridis")
    for k, r in enumerate(rs):
        style = "-" if r["corpus"] == "raw" else "--"
        col = cmap(k / max(1, len(rs) - 1))
        a.plot(DUPS, [r["exposure"][d] for d in DUPS], style, marker="o", ms=3, color=col, label=label(r))
        b.plot(DUPS, [100 * r[f"canary_extract_x{d}"] for d in DUPS], style, marker="o", ms=3, color=col)
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(DUPS, [str(d) for d in DUPS])
        ax.set_xlabel("Times a canary is inserted")
    a.axhline(0, color="#999", lw=0.6)
    a.set_ylabel("Exposure (nats/note)")
    b.set_ylabel("Canaries extracted (%)")
    a.legend(fontsize=5.8, loc="upper left", ncol=1)
    fig.savefig(ROOT / "figures" / "fig4_memorization.pdf")
    fig.savefig(ROOT / "figures" / "fig4_memorization.png")
    plt.close(fig)


if __name__ == "__main__":
    rs = rows()
    for r in rs:
        print(f"{label(r):32s} test {r['test_nll_per_note']:.3f}  exposure x1 {r['exposure'][1]:+.3f} x8 {r['exposure'][8]:+.3f} x32 {r['exposure'][32]:+.3f}"
              f"  extract max {r['max_extract']:.2f}  copying {r['copying']}")
    (ROOT / "paper" / "tismir" / "numbers" / "mem_table.tex").write_text(latex(rs))
    (ROOT / "results" / "13_memorization.json").write_text(json.dumps(
        [{k: v for k, v in r.items()} for r in rs], indent=1, default=float))
    figure(rs)
