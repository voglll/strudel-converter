"""
Extract drum samples from HPSS drum stem: short transients (30-100ms), per-type filtering.
The HPSS percussive component isolates transients from sustained tones.
"""
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import librosa, numpy as np, json

SR = 22050
DRUM_STEM = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\Toter Schmetterling_drums.wav")
BEAT = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\beat_samples")
SJSON = Path(r"c:\Users\micha\Desktop\strudel\strudel.json")

y, _ = librosa.load(DRUM_STEM, sr=SR, mono=True)
y = librosa.util.normalize(y[:int(10 * SR)])

def bp(data, lo, hi, order=4):
    nyq = SR / 2
    b, a = butter(order, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, data)

def hp(data, cutoff, order=4):
    nyq = SR / 2
    b, a = butter(order, cutoff / nyq, btype="highpass")
    return filtfilt(b, a, data)

# Detect onsets on drum stem
oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=256)
ons = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=256, backtrack=True, units="frames")
ons = sorted(set(ons))
ons = [o for i, o in enumerate(ons) if i == 0 or o - ons[i - 1] >= 2]

# Classify by band energy in drum stem
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
ke = S[(freqs >= 20) & (freqs <= 200)].mean(axis=0)
se = S[(freqs >= 300) & (freqs <= 5000)].mean(axis=0)
he = S[(freqs >= 6000)].mean(axis=0)

kicks_raw, snares_raw, hats_raw = [], [], []
kick_lens, snare_lens, hat_lens = [], [], []

for i, o in enumerate(ons):
    s0 = int(o * 256)
    s1 = int(ons[i + 1] * 256) if i + 1 < len(ons) else len(y)
    if s1 - s0 < SR // 50: continue
    
    lo, hi = max(0, o - 1), min(len(ke), o + 3)
    k = float(np.mean(ke[lo:hi]))
    s = float(np.mean(se[lo:hi]))
    h = float(np.mean(he[lo:hi]))
    mx = max(k, s, h)
    if mx < 0.001: continue

    # TAKE ONLY THE TRANSIENT — short window
    if k == mx:
        length = min(int(SR * 0.1), s1 - s0)  # max 100ms
        seg = y[s0:s0 + length]
        seg = bp(seg, 40, 180)
        kicks_raw.append(seg)
        kick_lens.append(length)
    elif s == mx:
        length = min(int(SR * 0.06), s1 - s0)  # max 60ms
        seg = y[s0:s0 + length]
        seg = bp(seg, 400, 6000)
        snares_raw.append(seg)
        snare_lens.append(length)
    else:
        length = min(int(SR * 0.03), s1 - s0)  # max 30ms
        seg = y[s0:s0 + length]
        seg = hp(seg, 7000)
        hats_raw.append(seg)
        hat_lens.append(length)

print(f"Kick: {len(kicks_raw)} hits, avg len: {np.mean(kick_lens)/SR*1000:.0f}ms")
print(f"Snare: {len(snares_raw)} hits, avg len: {np.mean(snare_lens)/SR*1000:.0f}ms")
print(f"Hat: {len(hats_raw)} hits, avg len: {np.mean(hat_lens)/SR*1000:.0f}ms")

def save(segs, name, pad_to=0.15):
    if not segs: return None
    # Pick the loudest segment as canonical
    best = max(segs, key=lambda s: float(np.max(np.abs(s))))
    # Pad to pad_to seconds
    target = int(pad_to * SR)
    if len(best) < target:
        best = np.pad(best, (0, target - len(best)))
    else:
        best = best[:target]
    # Fade out
    fo = min(int(SR * 0.01), len(best) // 4)
    if fo > 0:
        best = best.copy()
        best[-fo:] *= np.linspace(1, 0, fo)
    peak = float(np.max(np.abs(best))) if best.size else 1.0
    scaled = (best / max(1e-9, peak) * 32767 * 0.9).astype(np.int16)
    wavfile.write(str(BEAT / f"{name}.wav"), SR, scaled)
    print(f"  {name}.wav — {len(segs)} hits, {len(best)/SR*1000:.0f}ms")
    return name

samples = {}
for segs, name, pad in [(kicks_raw, "dr_kick", 0.12), (snares_raw, "dr_snare", 0.10), (hats_raw, "dr_hat", 0.06)]:
    r = save(segs, name, pad)
    if r:
        samples[r] = f"Toter%20Schmetterling_stems/beat_samples/{r}.wav"

# Update strudel.json
old = json.loads(SJSON.read_text())
for k, v in old.items():
    if k not in samples:
        samples[k] = v
SJSON.write_text(json.dumps({"_base": "https://raw.githubusercontent.com/voglll/strudel-converter/main/", **samples}, indent=2))
print(f"\nDone. {len(samples)} samples in strudel.json")
