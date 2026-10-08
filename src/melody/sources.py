"""Extract monophonic melodies from each corpus into Melody objects, cached as pickles.

Corpora used: Hooktheory (Sheet Sage release, official song-level splits), Essen Folksong
Collection (**kern), POP909 (MELODY track), Nottingham (jukedeck ABC, cleaned). The Session and
IrishMAN are excluded on licensing grounds (The Session's licence prohibits LLM use)."""
import glob
import gzip
import json
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .representation import Melody, REST

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "melodies"


def _cache(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    mels = build()
    path.write_bytes(pickle.dumps(mels))
    return mels


def hooktheory():
    def build():
        data = json.load(gzip.open(RAW / "hooktheory" / "Hooktheory.json.gz"))
        out = []
        for key, x in data.items():
            mel = x["annotations"].get("melody") or []
            if len(mel) < 8:
                continue
            mel = sorted(mel, key=lambda n: n["onset"])
            pitch = np.array([60 + 12 * n["octave"] + n["pitch_class"] for n in mel])
            onset = np.array([float(n["onset"]) for n in mel])
            dur = np.array([float(n["offset"] - n["onset"]) for n in mel])
            out.append(Melody(pitch, onset, dur, "hooktheory", key, x["split"].lower(),
                              {"artist": x["hooktheory"]["artist"], "song": x["hooktheory"]["song"]}))
        return out
    return _cache("hooktheory", build)


def _parse_music21(path):
    warnings.filterwarnings("ignore")
    from music21 import converter
    try:
        s = converter.parse(path)
    except Exception:
        return None
    part = s.parts[0] if hasattr(s, "parts") and len(s.parts) else s
    notes = [n for n in part.flatten().notes if len(n.pitches) and type(n).__name__ != "ChordSymbol"]
    if len(notes) < 8:
        return None
    pitch, onset, dur = [], [], []
    for n in notes:
        pitch.append(max(p.midi for p in n.pitches) if n.isChord else n.pitch.midi)
        onset.append(float(n.offset))
        dur.append(float(n.quarterLength))
    order = np.argsort(onset, kind="stable")
    return np.array(pitch)[order], np.array(onset)[order], np.array(dur)[order]


def _many(paths, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_parse_music21, paths, chunksize=16))


def essen():
    def build():
        paths = sorted(glob.glob(str(RAW / "essen" / "**" / "*.krn"), recursive=True))
        out = []
        for path, r in zip(paths, _many(paths)):
            if r is None:
                continue
            rel = Path(path).relative_to(RAW / "essen")
            region = rel.parts[1] if len(rel.parts) > 2 else ""
            out.append(Melody(*r, "essen", str(rel), "", {"region": region, "collection": rel.parts[-2]}))
        return out
    return _cache("essen", build)


def nottingham():
    """jukedeck cleaned ABC; one file per collection, many tunes per file."""
    def build():
        warnings.filterwarnings("ignore")
        from music21 import converter
        base = glob.glob(str(RAW / "nottingham_abc" / "nottingham-dataset-*" / "ABC_cleaned"))[0]
        out = []
        for path in sorted(glob.glob(base + "/*.abc")):
            text = Path(path).read_text(errors="ignore")
            chunks = ["X:" + c for c in text.split("\nX:")[1:]] if "\nX:" in text else [text]
            if text.startswith("X:"):
                chunks = ["X:" + c for c in ("\n" + text).split("\nX:")[1:]]
            for chunk in chunks:
                try:
                    s = converter.parseData(chunk, format="abc")
                except Exception:
                    continue
                notes = [n for n in s.flatten().notes if len(n.pitches) and type(n).__name__ != "ChordSymbol"]
                if len(notes) < 8:
                    continue
                pitch = np.array([max(p.midi for p in n.pitches) if n.isChord else n.pitch.midi for n in notes])
                onset = np.array([float(n.offset) for n in notes])
                dur = np.array([float(n.quarterLength) for n in notes])
                num = chunk.split("\n", 1)[0][2:].strip()
                out.append(Melody(pitch, onset, dur, "nottingham", f"{Path(path).stem}_{num}", "",
                                  {"collection": Path(path).stem}))
        return out
    return _cache("nottingham", build)


def _pop909_one(path):
    import mido
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    tpb = mid.ticks_per_beat
    for track in mid.tracks:
        if track.name.strip().upper() != "MELODY":
            continue
        t, active, notes = 0, {}, []
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = t
            elif msg.type in ("note_off", "note_on") and msg.note in active:
                start = active.pop(msg.note)
                notes.append((start / tpb, (t - start) / tpb, msg.note))
        if len(notes) < 8:
            return None
        notes.sort()
        on, du, pi = (np.array(x) for x in zip(*notes))
        return pi, on, du
    return None


def pop909():
    def build():
        paths = sorted(glob.glob(str(RAW / "pop909" / "**" / "POP909" / "*" / "[0-9][0-9][0-9].mid"), recursive=True))
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(_pop909_one, paths, chunksize=16))
        return [Melody(r[0], r[1], r[2], "pop909", Path(p).stem, "", {}) for p, r in zip(paths, res) if r is not None]
    return _cache("pop909", build)


def _pdmx_one(path):
    """Skyline melody: in the most melodic non-drum track (highest mean pitch among tracks with
    16+ notes), keep the highest note at each onset."""
    import mido
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    tpb = mid.ticks_per_beat or 480
    best, best_mean = None, -1
    for track in mid.tracks:
        t, active, notes = 0, {}, []
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.channel != 9:
                active[msg.note] = t
            elif msg.type in ("note_off", "note_on") and getattr(msg, "note", None) in active:
                s = active.pop(msg.note)
                notes.append((s, t - s, msg.note))
        if len(notes) >= 16:
            m = np.mean([n[2] for n in notes])
            if m > best_mean:
                best, best_mean = notes, m
    if best is None:
        return None
    best.sort(key=lambda n: (n[0], -n[2]))
    on, du, pi = [], [], []
    last = None
    for s, d, p in best:
        if s == last:
            continue
        on.append(s / tpb); du.append(max(d, 1) / tpb); pi.append(p)
        last = s
    return np.array(pi, np.int16), np.array(on, np.float32), np.array(du, np.float32)


def pdmx(limit=None):
    """Melodies from PDMX (public domain / CC0), the no-licence-conflict subset. Each Melody keeps
    PDMX's own `subset:deduplicated` flag in meta for comparison with our audit."""
    import pandas as pd

    def build():
        meta = pd.read_csv(RAW / "pdmx" / "PDMX.csv", low_memory=False)
        meta = meta[meta["subset:no_license_conflict"] & meta["mid"].notna()]
        if limit:
            meta = meta.iloc[:limit]
        paths = [str(RAW / "pdmx" / p[2:]) for p in meta["mid"]]
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(_pdmx_one, paths, chunksize=64))
        out = []
        for (_, row), r in zip(meta.iterrows(), res):
            if r is None:
                continue
            out.append(Melody(r[0], r[1], r[2], "pdmx", row["mid"][2:], "",
                              {"dedup": bool(row["subset:deduplicated"]), "title": str(row["title"])[:80],
                               "genres": str(row["genres"]), "rating": float(row["rating"]) if row["rating"] == row["rating"] else None}))
        return out
    return _cache("pdmx" if not limit else f"pdmx_{limit}", build)
