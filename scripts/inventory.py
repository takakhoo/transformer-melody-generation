"""Count tunes per dataset under data/raw and print a few Nottingham examples.

The Session's LICENSE.md (since 2025-10-08) prohibits processing its material
with LLM tools, so by default only Session metadata is printed. Pass
--show-session-abc to print the ABC bodies yourself.
"""
import argparse
import collections
import csv
import glob
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
RAW = os.path.normpath(RAW)
csv.field_size_limit(10**9)


def p(*a):
    print(*a, flush=True)


def one(pattern):
    hits = sorted(h for h in glob.glob(os.path.join(RAW, pattern)) if not h.endswith(".zip"))
    return hits[0] if hits else None


class SafeUnpickler(pickle.Unpickler):
    # Pickles fetched from the web can run code on load; only allow numpy scalar/array rebuilds.
    ALLOWED = {
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        import numpy
        if module == "numpy":
            return getattr(numpy, name)
        core = getattr(numpy, "_core", None) or numpy.core
        return getattr(core.multiarray, name)


def split_abc(text):
    tunes, cur = [], []
    for line in text.splitlines():
        if re.match(r"^X:\s*\d+", line):
            if cur:
                tunes.append("\n".join(cur).strip())
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        tunes.append("\n".join(cur).strip())
    return tunes


def tar_count(path, suffixes):
    n = collections.Counter()
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if m.isfile():
                low = m.name.lower()
                for s in suffixes:
                    if low.endswith(s):
                        n[s] += 1
                n["_files"] += 1
    return n


def thesession(show_abc):
    p("\n## The Session (adactio/TheSession-data)")
    path = one("thesession/TheSession-data-*/csv/tunes.csv")
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    tunes = {}
    for r in rows:
        tunes.setdefault(r["tune_id"], r)
    types = collections.Counter(r["type"] for r in tunes.values())
    p(f"settings (rows in tunes.csv): {len(rows)}")
    p(f"unique tunes (tune_id): {len(tunes)}")
    p("tunes by type:", dict(types.most_common()))
    p("example tunes (first setting of tune_id 1, 2, 3):")
    for tid in ["1", "2", "3"]:
        r = tunes.get(tid)
        if not r:
            continue
        p(f"  tune_id={r['tune_id']} setting_id={r['setting_id']} name={r['name']!r} "
          f"type={r['type']} meter={r['meter']} mode={r['mode']} abc_chars={len(r['abc'])}")
        if show_abc:
            p(r["abc"])


def nottingham():
    p("\n## Nottingham Music Database")
    base = one("nottingham_abc/nottingham-dataset-*")
    for sub in ["ABC_original", "ABC_cleaned"]:
        files = sorted(glob.glob(os.path.join(base, sub, "*.abc")))
        n = sum(len(split_abc(open(f, encoding="latin-1").read())) for f in files)
        p(f"jukedeck {sub}: {len(files)} files, {n} tunes")
    mids = glob.glob(os.path.join(base, "MIDI", "**", "*.mid"), recursive=True)
    by = collections.Counter(os.path.relpath(os.path.dirname(f), os.path.join(base, "MIDI")) for f in mids)
    p(f"jukedeck MIDI: {len(mids)} .mid files; by dir (. = full arrangement): {dict(by)}")
    z = zipfile.ZipFile(os.path.join(RAW, "nottingham_abc", "nottingham_database.zip"))
    n = 0
    for name in z.namelist():
        if name.endswith(".abc"):
            n += len(split_abc(z.read(name).decode("latin-1")))
    p(f"Seymour Shlien edit (nottingham_database.zip): {n} tunes")
    jigs = split_abc(open(os.path.join(base, "ABC_cleaned", "jigs.abc"), encoding="latin-1").read())
    p("example tunes (jukedeck ABC_cleaned/jigs.abc, first 3):")
    for t in jigs[:3]:
        p("```abc\n" + t + "\n```")


def boulanger():
    p("\n## Boulanger-Lewandowski et al. 2012 piano-roll splits")
    d = os.path.join(RAW, "boulanger2012")
    for f in ["wayback_JSB Chorales.pickle", "wayback_Piano-midi.de.pickle"]:
        fp = os.path.join(d, f)
        if os.path.exists(fp):
            data = SafeUnpickler(open(fp, "rb"), encoding="latin1").load()
            p(f"{f}: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()))
    try:
        from scipy.io import loadmat
    except ImportError:
        p("scipy missing, skipping .mat files")
        return
    for fp in sorted(glob.glob(os.path.join(d, "tcn_*.mat"))):
        m = loadmat(fp)
        keys = [k for k in m if not k.startswith("__")]
        p(f"{os.path.basename(fp)}: " + ", ".join(f"{k}={m[k].shape[1] if m[k].ndim == 2 else m[k].shape}" for k in keys))
    z = os.path.join(d, "wayback_Nottingham.zip")
    if os.path.exists(z):
        c = collections.Counter(n.split("/")[1] for n in zipfile.ZipFile(z).namelist() if n.endswith(".mid"))
        p(f"wayback_Nottingham.zip (source MIDI): {dict(c)}")


def essen():
    p("\n## Essen Folksong Collection (**kern)")
    base = one("essen/essen-folksong-collection-*")
    files = glob.glob(os.path.join(base, "**", "*.krn"), recursive=True)
    by = collections.Counter(os.path.relpath(f, base).split(os.sep)[0] for f in files)
    p(f".krn files: {len(files)}; by top-level dir: {dict(by.most_common())}")


def pdmx():
    p("\n## PDMX")
    d = os.path.join(RAW, "pdmx")
    fp = os.path.join(d, "PDMX.csv")
    r = csv.reader(open(fp, newline="", encoding="utf-8"))
    h = next(r)
    si = [i for i, c in enumerate(h) if c.startswith("subset:")]
    li = h.index("license")
    n, lic, sub = 0, collections.Counter(), collections.Counter()
    for row in r:
        n += 1
        lic[row[li]] += 1
        for i in si:
            sub[h[i]] += row[i] == "True"
    p(f"PDMX.csv rows (scores): {n}; license field: {dict(lic)}")
    p("subset flags:", dict(sub))
    for f, suf in [("mxl.tar.gz", [".mxl"]), ("mid.tar.gz", [".mid"]), ("data.tar.gz", [".json"])]:
        fp = os.path.join(d, f)
        if os.path.exists(fp):
            try:
                p(f"{f}: {dict(tar_count(fp, suf))}")
            except Exception as e:
                p(f"{f}: unreadable ({e})")
        else:
            p(f"{f}: not downloaded")


def lakh():
    p("\n## Lakh MIDI Dataset")
    d = os.path.join(RAW, "lakh")
    for f in ["lmd_full.tar.gz", "lmd_matched.tar.gz", "clean_midi.tar.gz"]:
        fp = os.path.join(d, f)
        if os.path.exists(fp):
            c = tar_count(fp, [".mid"])
            p(f"{f}: {c['.mid']} .mid files ({c['_files']} files total)")
    m = json.load(open(os.path.join(d, "md5_to_paths.json")))
    p(f"md5_to_paths.json: {len(m)} md5 entries")
    s = json.load(open(os.path.join(d, "match_scores.json")))
    p(f"match_scores.json: {len(s)} MSD tracks, {sum(len(v) for v in s.values())} (track, midi) pairs")


def pop909():
    p("\n## POP909")
    base = one("pop909/POP909-Dataset-*/POP909")
    songs = sorted(x for x in os.listdir(base) if x.isdigit())
    mids = glob.glob(os.path.join(base, "*", "*.mid"))
    vers = glob.glob(os.path.join(base, "*", "versions", "*.mid"))
    p(f"songs: {len(songs)}; main .mid: {len(mids)}; alternate versions .mid: {len(vers)}")
    try:
        import mido
        names = collections.Counter()
        for f in mids:
            names.update(t.name for t in mido.MidiFile(f).tracks if t.name)
        p("track names across main .mid:", dict(names))
    except ImportError:
        pass


def irishman():
    p("\n## IrishMAN")
    d = os.path.join(RAW, "irishman")
    for f in ["train.json", "validation.json", "leadsheet_ids.json", "variation_ids.json"]:
        obj = json.load(open(os.path.join(d, f)))
        if isinstance(obj, dict) and all(isinstance(v, list) for v in obj.values()):
            p(f"{f}: " + ", ".join(f"{k}={len(v)}" for k, v in obj.items()))
        else:
            p(f"{f}: {len(obj)} entries")
    for f in ["irishman-midi.zip", "irishman-xml.zip"]:
        names = zipfile.ZipFile(os.path.join(d, f)).namelist()
        ext = collections.Counter(os.path.splitext(n)[1].lower() for n in names if not n.endswith("/"))
        p(f"{f}: {dict(ext)}")


def hooktheory():
    p("\n## Hooktheory (Sheet Sage release)")
    d = json.load(gzip.open(os.path.join(RAW, "hooktheory", "Hooktheory.json.gz")))
    split = collections.Counter(v["split"] for v in d.values())
    mel = sum("MELODY" in v["tags"] for v in d.values())
    p(f"annotations: {len(d)}; by split: {dict(split)}; with MELODY tag: {mel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-session-abc", action="store_true")
    ap.add_argument("--skip-big", action="store_true", help="skip tar listings of Lakh/PDMX")
    a = ap.parse_args()
    p(f"# Inventory of {RAW}")
    thesession(a.show_session_abc)
    nottingham()
    boulanger()
    essen()
    if not a.skip_big:
        pdmx()
        lakh()
    pop909()
    irishman()
    hooktheory()


if __name__ == "__main__":
    main()
