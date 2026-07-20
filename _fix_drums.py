"""Re-extract drums with per-type bandpass filtering to clean up snares."""
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import librosa, numpy as np, json

SR = 22050; DUR = 0.3
STEMS = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems")
BEAT = STEMS / "beat_samples"
SJSON = Path(r"c:\Users\micha\Desktop\strudel\strudel.json")

y, _ = librosa.load(STEMS / "Toter Schmetterling_drums.wav", sr=SR, mono=True)
y = librosa.util.normalize(y[:int(10 * SR)])

# Onsets
oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
ons = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=512, backtrack=True, units="frames")
ons = sorted(set(ons))
ons = [o for i, o in enumerate(ons) if i == 0 or o - ons[i - 1] >= 2]

# Classify each hit by band energy
S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
freqs = librosa.fft_frequencies(sr=SR, n_fft=4096)
ke = S[(freqs >= 20) & (freqs <= 160)].mean(axis=0)
se = S[(freqs >= 160) & (freqs <= 4500)].mean(axis=0)
he = S[(freqs >= 4500)].mean(axis=0)

def bandpass(data, lo, hi, order=4):
    nyq = SR / 2
    b, a = butter(order, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, data)

kick_segs, snare_segs, hat_segs = [], [], []

for i, o in enumerate(ons):
    s0 = int(o * 512)
    s1 = int(ons[i + 1] * 512) if i + 1 < len(ons) else len(y)
    if s1 - s0 < SR // 25: continue
    
    lo, hi = max(0, o - 1), min(len(ke), o + 4)
    k = float(np.mean(ke[lo:hi])); s = float(np.mean(se[lo:hi])); h = float(np.mean(he[lo:hi]))
    mx = max(k, s, h)
    if mx < 0.001: continue
    
    seg = y[s0:s1]

    if k == mx:
        # Kick: bandpass 30-160Hz — keep sub, remove bleed
        seg = bandpass(seg, 30, 160)
        kick_segs.append(seg)
    elif s == mx:
        # Snare: bandpass 200-5000Hz — remove kick rumble
        seg = bandpass(seg, 200, 5000)
        snare_segs.append(seg)
    else:
        # Hat: highpass 5000Hz
        seg = bandpass(seg, 5000, 10000)
        hat_segs.append(seg)

def make_sample(segs, name):
    if not segs:
        print(f"  {name}: NO SEGMENTS — skipping")
        return None
    best = max(segs, key=lambda s: len(s))
    target = int(DUR * SR)
    if len(best) >= target: best = best[:target]
    else: best = np.pad(best, (0, target - len(best)))
    fi = min(int(SR * 0.005), len(best) // 4)
    fo = min(int(SR * 0.02), len(best) // 4)
    if fi > 0: best[:fi] *= np.linspace(0, 1, fi)
    if fo > 0: best[-fo:] *= np.linspace(1, 0, fo)
    peak = float(np.max(np.abs(best))) if best.size else 1.0
    scaled = (best / max(1e-9, peak) * 32767 * 0.9).astype(np.int16)
    wavfile.write(str(BEAT / f"{name}.wav"), SR, scaled)
    print(f"  {name}.wav — {len(segs)} hits, peak={peak:.2f}")
    return name

print("Re-extracting drums with per-type filtering...")
samples = {}
for segs, name in [(kick_segs, "dr_kick"), (snare_segs, "dr_snare"), (hat_segs, "dr_hat")]:
    result = make_sample(segs, name)
    if result:
        samples[result] = f"Toter%20Schmetterling_stems/beat_samples/{result}.wav"

# Keep existing harmony, texture, bass mappings
old_json = json.loads(SJSON.read_text())
for k, v in old_json.items():
    if k not in samples:
        samples[k] = v

SJSON.write_text(json.dumps({"_base": "https://raw.githubusercontent.com/voglll/strudel-converter/main/", **samples}, indent=2))
print(f"\nDone. {len(samples)} samples in strudel.json")
