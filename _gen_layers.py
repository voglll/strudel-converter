"""Generate layered stack() pattern — each stem in its own stack() block."""
from pathlib import Path
from collections import Counter
import librosa, numpy as np

SR = 22050; NFFT = 4096; HOP = 512; FOCUS = 10.0; TEMPO = 155.88; BD = 60.0 / TEMPO
STEMS = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems")
BEAT = STEMS / "beat_samples"
BASS = STEMS / "bass_samples"
OUT = Path(r"c:\Users\micha\Desktop\strudel\BASS_LINE.txt")

def hz2n(hz):
    if np.isnan(hz) or hz <= 0: return "~"
    return librosa.hz_to_note(float(hz), octave=True, unicode=False).replace("#", "s")

def getset(d, p):
    return {f.stem for f in d.glob(f"{p}*.wav")}

def extract(y, sample_set, prefix):
    oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    ons = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=HOP, backtrack=True, units="frames")
    ons = sorted(set(ons))
    ons = [o for i, o in enumerate(ons) if i == 0 or o - ons[i - 1] >= 2]
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    f0, _, _ = librosa.pyin(y, fmin=30, fmax=2000, sr=SR, frame_length=NFFT, hop_length=HOP)
    pn = [hz2n(f) for f in f0]
    ke = S[(freqs >= 20) & (freqs <= 160)].mean(axis=0)
    se = S[(freqs >= 160) & (freqs <= 4500)].mean(axis=0)
    he = S[(freqs >= 4500)].mean(axis=0)
    seq = []
    for i, o in enumerate(ons):
        lo, hi = max(0, o - 1), min(len(pn), o + 4)
        if prefix == "dr":
            k = float(np.mean(ke[lo:hi])); s = float(np.mean(se[lo:hi])); h = float(np.mean(he[lo:hi]))
            mx = max(k, s, h)
            if mx < 0.001: continue
            label = "dr_kick" if k == mx else ("dr_snare" if s == mx else "dr_hat")
        else:
            w = [n for n in pn[lo:hi] if n != "~"]
            if not w: continue
            label = f"{prefix}_{Counter(w).most_common(1)[0][0]}"
        if label in sample_set: seq.append((o * HOP / SR, label))
    return seq

def pat(seq, spb=4):
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

# Load
yb = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_bass.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
yd = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_drums.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
yh = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_harmony.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
yt = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_texture.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])

bset = getset(BASS, "bass_"); dset = getset(BEAT, "dr_"); hset = getset(BEAT, "hm_"); tset = getset(BEAT, "tx_")
bseq = extract(yb, bset, "bass"); dseq = extract(yd, dset, "dr"); hseq = extract(yh, hset, "hm"); tseq = extract(yt, tset, "tx")

bp = pat(bseq); dp = pat(dseq); hp = pat(hseq); tp = pat(tseq)
cps = TEMPO / 60.0

code = f"""samples('github:voglll/strudel-converter')
setcps({cps:.4f})

stack(
  // bass
  stack(`{bp}`).gain(0.9).lpf(500),
  // drums
  stack(`{dp}`).gain(0.85).room(0.05),
  // harmony
  stack(`{hp}`).gain(0.4).room(0.2).lpf(3000),
  // texture
  stack(`{tp}`).gain(0.2).room(0.3).hpf(2000),
)
"""

OUT.write_text(code, encoding="utf-8")
print(f"Wrote {OUT} ({len(code)} bytes)")
print(code[:300])
