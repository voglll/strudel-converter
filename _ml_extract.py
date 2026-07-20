"""Extract from Demucs ML stems + generate BEAT_v3.txt"""
from pathlib import Path
from collections import Counter
from scipy.io import wavfile
import librosa, numpy as np, json

SR = 22050; HOP = 256; FOCUS = 10.0; TEMPO = 155.88; BD = 60.0 / TEMPO; V = "v3"
STEMS = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems")
BEAT = STEMS / "beat_samples"
BASS_S = STEMS / "bass_samples"
SJSON = Path(r"c:\Users\micha\Desktop\strudel\strudel.json")
OUT = Path(r"c:\Users\micha\Desktop\strudel\BEAT_v3.txt")

for f in BEAT.glob("dr_*.wav"):
    if "_v" not in f.stem: f.unlink()

def hz2n(hz):
    if np.isnan(hz) or hz <= 0: return "~"
    return librosa.hz_to_note(float(hz), octave=True, unicode=False).replace("#", "s")

def onsets(y):
    oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    o = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=HOP, backtrack=True, units="frames")
    o = sorted(set(o))
    return [x for i, x in enumerate(o) if i == 0 or x - o[i - 1] >= 2]

print("DRUMS (Demucs ML stem)...")
yd = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_drums_ml.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])

# Band-specific onset detection on ML stem
def bp(data, lo, hi):
    from scipy.signal import butter, filtfilt
    nyq = SR/2; b, a = butter(4, [lo/nyq, min(hi, nyq*0.99)/nyq], btype="band")
    return filtfilt(b, a, data)

y_k = bp(yd, 30, 200)
y_s = bp(yd, 400, 6000)
y_h = bp(yd, 7000, 10000)

k_ons = set(onsets(y_k))
s_ons = set(onsets(y_s))
h_ons = set(onsets(y_h))

kicks, snares, hats = [], [], []
drum_seq = []
used = set()
for o in sorted(k_ons | s_ons | h_ons):
    if o in used or o < 2: continue
    s0 = int(o * HOP); s1 = min(s0 + int(0.08 * SR), len(yd))
    seg = yd[s0:s1]
    ik, is_, ih = o in k_ons, o in s_ons, o in h_ons
    if is_ and not ik:
        snares.append(seg); label = f"dr_snare_{V}"
    elif ih and not ik and not is_:
        hats.append(seg); label = f"dr_hat_{V}"
    elif ik:
        kicks.append(seg); label = f"dr_kick_{V}"
    elif is_:
        snares.append(seg); label = f"dr_snare_{V}"
    else:
        hats.append(seg); label = f"dr_hat_{V}"
    drum_seq.append((o * HOP / SR, label))
    used.add(o)
print(f"  Kick:{len(kicks)} Snare:{len(snares)} Hat:{len(hats)} Events:{len(drum_seq)}")

print("BASS (Demucs ML stem)...")
yb = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_bass_ml.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
f0, _, _ = librosa.pyin(yb, fmin=30, fmax=500, sr=SR, frame_length=4096, hop_length=HOP)
pn = [hz2n(f) for f in f0]
b_ons = onsets(yb)
bass_seq = []; bass_segs = {}
for o in b_ons:
    lo, hi = max(0, o - 1), min(len(pn), o + 4)
    w = [n for n in pn[lo:hi] if n != "~"]
    if not w: continue
    note = Counter(w).most_common(1)[0][0]; key = f"bass_ml_{note}"
    s0 = int(o * HOP); s1 = min(s0 + int(0.25 * SR), len(yb))
    bass_segs.setdefault(key, []).append(yb[s0:s1])
    bass_seq.append((o * HOP / SR, key))
print(f"  Bass events: {len(bass_seq)} notes: {len(bass_segs)}")

def save(segs, name, dur=0.15):
    if not segs: return
    best = max(segs, key=lambda s: float(np.max(np.abs(s))) if s.size else 0)
    t = int(dur * SR)
    if len(best) >= t: best = best[:t]
    else: best = np.pad(best, (0, t - len(best)))
    fo = min(int(SR * 0.01), len(best) // 4)
    if fo > 0: best[-fo:] *= np.linspace(1, 0, fo)
    p = float(np.max(np.abs(best))) if best.size else 1.0
    s = (best / max(1e-9, p) * 32767 * 0.9).astype(np.int16)
    wavfile.write(str(BEAT / f"{name}.wav"), SR, s)
    print(f"  {name}.wav")

print("Saving samples...")
save(kicks, f"dr_kick_{V}", 0.12)
save(snares, f"dr_snare_{V}", 0.10)
save(hats, f"dr_hat_{V}", 0.06)
for name, segs in bass_segs.items(): save(segs, name, 0.25)

smap = {"_base": "https://raw.githubusercontent.com/voglll/strudel-converter/main/"}
for f in list(BEAT.glob(f"*_{V}.wav")) + list(BASS_S.glob("bass_ml_*.wav")):
    folder = "beat_samples" if "bass_ml_" not in f.stem else "bass_samples"
    smap[f.stem] = f"Toter%20Schmetterling_stems/{folder}/{f.name}"
SJSON.write_text(json.dumps(smap, indent=2))

def seq_to_pat(seq, spb=4):
    nb = int(FOCUS / BD) + 1; ns = nb * spb; sd = BD / spb
    grid = [[] for _ in range(ns)]
    for t, k in seq:
        si = int(t / sd)
        if 0 <= si < ns: grid[si].append(k)
    bars = []
    for bs in range(0, ns, 16):
        st = []
        for si in range(bs, min(bs + 16, ns)):
            ids = grid[si]
            st.append("~" if not ids else (ids[0] if len(ids) == 1 else "[" + " ".join(ids) + "]"))
        bars.append("[" + " ".join(st) + "]")
    return " ".join(bars)

bp = seq_to_pat(bass_seq); dp = seq_to_pat(drum_seq)
cps = TEMPO / 60.0
code = f"samples('github:voglll/strudel-converter')\nsetcps({cps:.4f})\n\nstack(\n  // bass (Demucs ML)\n  stack(`{bp}`).gain(0.9).lpf(500),\n  // drums (Demucs ML)\n  stack(`{dp}`).gain(0.85).room(0.05),\n)\n"
OUT.write_text(code)
print(f"\n✅ {OUT.name} ({len(code)} bytes)")
print(code[:400])
