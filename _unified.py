"""Unified: band-specific drum detection + pattern generation + sample creation."""
from pathlib import Path
from collections import Counter
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import librosa, numpy as np, json

SR = 22050; NFFT = 4096; HOP = 512; FOCUS = 10.0; TEMPO = 155.88; BD = 60.0 / TEMPO; DUR = 0.3
ORIG = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling.mp3")
STEMS = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems")
BEAT = STEMS / "beat_samples"
BASS = STEMS / "bass_samples"
SJSON = Path(r"c:\Users\micha\Desktop\strudel\strudel.json")
OUT = Path(r"c:\Users\micha\Desktop\strudel\BASS_LINE.txt")

def bp(data, lo, hi):
    nyq = SR / 2; b, a = butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, data)

def hz2n(hz):
    if np.isnan(hz) or hz <= 0: return "~"
    return librosa.hz_to_note(float(hz), octave=True, unicode=False).replace("#", "s")

def onset_frames(y_band):
    oe = librosa.onset.onset_strength(y=y_band, sr=SR, hop_length=HOP)
    return sorted(set(librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=HOP, backtrack=True, units="frames")))

# Load
y_orig, _ = librosa.load(ORIG, sr=SR, mono=True)
y_orig = librosa.util.normalize(y_orig[:int(FOCUS * SR)])
yb = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_bass.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
yh = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_harmony.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
yt = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_texture.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])

# === DRUMS: band-specific onset detection from ORIGINAL ===
print("Drums: band-specific onset detection from original...")
y_kick_bp = bp(y_orig, 30, 160)
y_snare_bp = bp(y_orig, 300, 4000)
y_hat_bp = bp(y_orig, 5000, 10000)

kick_ons = set(onset_frames(y_kick_bp))
snare_ons = set(onset_frames(y_snare_bp))
hat_ons = set(onset_frames(y_hat_bp))

# Assign non-overlapping
used = set()
drum_seq = []  # (time, label)
kicks, snares, hats = [], [], []

for o in sorted(kick_ons | snare_ons | hat_ons):
    if o in used or o < 2: continue
    s0 = int(o * HOP); s1 = min(s0 + int(DUR * SR), len(y_orig)); s0 = max(0, s1 - int(DUR * SR))
    seg = y_orig[s0:s1]
    in_k, in_s, in_h = o in kick_ons, o in snare_ons, o in hat_ons

    if in_s and not in_k:
        snares.append(bp(seg, 200, 5000)); label = "dr_snare"
    elif in_h and not in_k and not in_s:
        hats.append(bp(seg, 5000, 10000)); label = "dr_hat"
    elif in_k:
        kicks.append(bp(seg, 30, 160)); label = "dr_kick"
    elif in_s:
        snares.append(bp(seg, 200, 5000)); label = "dr_snare"
    else:
        hats.append(bp(seg, 5000, 10000)); label = "dr_hat"
    drum_seq.append((o * HOP / SR, label))
    used.add(o)

print(f"  Kick: {len(kicks)}, Snare: {len(snares)}, Hat: {len(hats)}, Events: {len(drum_seq)}")

# === BASS: pyin from bass stem ===
print("Bass: pyin + onset detection...")
f0_b, _, _ = librosa.pyin(yb, fmin=30, fmax=500, sr=SR, frame_length=NFFT, hop_length=HOP)
bass_names = [hz2n(f) for f in f0_b]
bass_ons = onset_frames(yb)
bass_ons = [o for i, o in enumerate(bass_ons) if i == 0 or o - bass_ons[i - 1] >= 2]

bass_set = {f.stem for f in BASS.glob("bass_*.wav")}
bass_seq = []
for o in bass_ons:
    lo, hi = max(0, o - 1), min(len(bass_names), o + 4)
    w = [n for n in bass_names[lo:hi] if n != "~"]
    if not w: continue
    note = Counter(w).most_common(1)[0][0]
    key = f"bass_{note}"
    if key in bass_set:
        bass_seq.append((o * HOP / SR, key))

# === HARMONY + TEXTURE: pyin from stems ===
def melodic_seq(y, prefix, sample_dir):
    sample_set = {f.stem for f in sample_dir.glob(f"{prefix}_*.wav")}
    f0, _, _ = librosa.pyin(y, fmin=30, fmax=2000, sr=SR, frame_length=NFFT, hop_length=HOP)
    names = [hz2n(f) for f in f0]
    ons = onset_frames(y)
    ons = [o for i, o in enumerate(ons) if i == 0 or o - ons[i - 1] >= 2]
    seq = []
    for o in ons:
        lo, hi = max(0, o - 1), min(len(names), o + 4)
        w = [n for n in names[lo:hi] if n != "~"]
        if not w: continue
        note = Counter(w).most_common(1)[0][0]
        key = f"{prefix}_{note}"
        if key in sample_set:
            seq.append((o * HOP / SR, key))
    return seq

harmony_seq = melodic_seq(yh, "hm", BEAT)
texture_seq = melodic_seq(yt, "tx", BEAT)

print(f"Bass: {len(bass_seq)}, Harmony: {len(harmony_seq)}, Texture: {len(texture_seq)}")

# === SAVE DRUM SAMPLES ===
print("Saving drum samples...")
def save(segs, name):
    if not segs: return
    best = max(segs, key=lambda s: len(s))
    t = int(DUR * SR)
    if len(best) >= t: best = best[:t]
    else: best = np.pad(best, (0, t - len(best)))
    fi = min(int(SR * 0.005), len(best) // 4); fo = min(int(SR * 0.02), len(best) // 4)
    if fi > 0: best[:fi] *= np.linspace(0, 1, fi)
    if fo > 0: best[-fo:] *= np.linspace(1, 0, fo)
    p = float(np.max(np.abs(best))) if best.size else 1.0
    s = (best / max(1e-9, p) * 32767 * 0.9).astype(np.int16)
    wavfile.write(str(BEAT / f"{name}.wav"), SR, s)

save(kicks, "dr_kick")
save(snares, "dr_snare")
save(hats, "dr_hat")

# Update strudel.json with all samples
smap = {"_base": "https://raw.githubusercontent.com/voglll/strudel-converter/main/"}
for d, prefix in [(BEAT, "dr_"), (BEAT, "hm_"), (BEAT, "tx_"), (BASS, "bass_")]:
    for f in d.glob(f"{prefix}*.wav"):
        smap[f.stem] = f"Toter%20Schmetterling_stems/{'bass' if 'bass' in str(d) else 'beat'}_samples/{f.name}"
SJSON.write_text(json.dumps(smap, indent=2))

# === GENERATE PATTERN ===
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
            if not ids: st.append("~")
            elif len(ids) == 1: st.append(ids[0])
            else: st.append("[" + " ".join(ids) + "]")
        bars.append("[" + " ".join(st) + "]")
    return " ".join(bars)

bp_s = seq_to_pat(bass_seq)
dp_s = seq_to_pat(drum_seq)
hp_s = seq_to_pat(harmony_seq)
tp_s = seq_to_pat(texture_seq)
cps = TEMPO / 60.0

code = f"""samples('github:voglll/strudel-converter')
setcps({cps:.4f})

stack(
  // bass
  stack(`{bp_s}`).gain(0.9).lpf(500),
  // drums
  stack(`{dp_s}`).gain(0.85).room(0.05),
  // harmony
  stack(`{hp_s}`).gain(0.4).room(0.2).lpf(3000),
  // texture
  stack(`{tp_s}`).gain(0.2).room(0.3).hpf(2000),
)
"""
OUT.write_text(code, encoding="utf-8")
print(f"\nWrote {OUT} ({len(code)} bytes)")
print("Done!")
