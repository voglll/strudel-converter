"""
Re-extract bass samples with noise gate + detect second bass layer.
Outputs: bass1_*.wav (dominant) and bass2_*.wav (second layer).
"""
from pathlib import Path
from collections import Counter
from scipy.io import wavfile
from scipy.signal import find_peaks, butter, filtfilt
import librosa
import numpy as np
import json

BASS_WAV = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\Toter Schmetterling_bass.wav")
SAMPLES_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\bass_samples")
STRUDEL_JSON = Path(r"c:\Users\micha\Desktop\strudel\strudel.json")
SR = 22050
FOCUS_SEC = 15.0
N_FFT = 4096
HOP = 512

# Clean old samples
for f in SAMPLES_DIR.glob("*.wav"):
    f.unlink()
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading...")
y_full, _ = librosa.load(BASS_WAV, sr=SR, mono=True)
y = librosa.util.normalize(y_full[:int(FOCUS_SEC * SR)])

# Noise gate: apply lowpass to reduce HPSS noise
print("Applying noise gate (lowpass 250Hz)...")
nyq = SR / 2
b, a = butter(4, 250 / nyq, btype="lowpass")
y_clean = filtfilt(b, a, y)
y_clean = librosa.util.normalize(y_clean)

# -----------------------------------------------------------------------
# Multi-pitch detection per frame
# -----------------------------------------------------------------------
print("Multi-pitch analysis...")
stft = np.abs(librosa.stft(y_clean, n_fft=N_FFT, hop_length=HOP))
freqs_stft = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

# Onset detection
onset_env = librosa.onset.onset_strength(y=y_clean, sr=SR, hop_length=HOP)
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env, sr=SR, hop_length=HOP, backtrack=True, units="frames",
)

bass_mask = (freqs_stft >= 30) & (freqs_stft <= 500)
freqs_bass = freqs_stft[bass_mask]

# Per frame: detect dominant + second pitch
frame_pitches: list[tuple[float | None, float | None]] = []  # (dominant_hz, second_hz)

for frame in range(stft.shape[1]):
    col = stft[bass_mask, frame]
    if np.max(col) < 0.001:
        frame_pitches.append((None, None))
        continue
    peaks, _ = find_peaks(col, height=np.max(col) * 0.05, distance=4)
    if len(peaks) < 1:
        frame_pitches.append((None, None))
        continue
    sorted_idx = peaks[np.argsort(col[peaks])[::-1]]
    
    f1 = float(freqs_bass[sorted_idx[0]])
    
    f2 = None
    if len(sorted_idx) >= 2:
        f2_candidate = float(freqs_bass[sorted_idx[1]])
        ratio = f2_candidate / f1
        nearest_harmonic = min([1.5, 2.0, 2.5, 3.0, 4.0], key=lambda h: abs(ratio - h))
        if abs(ratio - nearest_harmonic) > 0.15:
            f2 = f2_candidate
    
    frame_pitches.append((f1, f2))

# -----------------------------------------------------------------------
# Segment into note events
# -----------------------------------------------------------------------
def hz_to_name(hz):
    if hz is None or hz <= 0:
        return "~"
    return librosa.hz_to_note(float(hz), octave=True, unicode=False)

# Use onset frames + pitch changes to segment
event_frames = sorted(set(onset_frames))
min_gap = 2
filtered = []
for f in event_frames:
    if not filtered or f - filtered[-1] >= min_gap:
        filtered.append(f)
event_frames = filtered

event_samples = [int(f * HOP) for f in event_frames]

# Collect segments for bass1 and bass2
samples_b1: dict[str, list[np.ndarray]] = {}
samples_b2: dict[str, list[np.ndarray]] = {}
sequence_b1: list[tuple[float, str]] = []
sequence_b2: list[tuple[float, str]] = []

for i, ef in enumerate(event_frames):
    lo = max(0, ef - 1)
    hi = min(len(frame_pitches), ef + 4)
    
    # Dominant note
    names1 = [hz_to_name(fp[0]) for fp in frame_pitches[lo:hi] if fp[0] is not None]
    if names1:
        note1 = Counter(names1).most_common(1)[0][0]
        start = event_samples[i]
        end = event_samples[i + 1] if i + 1 < len(event_samples) else len(y_clean)
        if end - start >= SR // 20:
            seg = y_clean[start:end]
            samples_b1.setdefault(note1, []).append(seg)
            sequence_b1.append((start / SR, note1))
    
    # Second bass
    names2 = [hz_to_name(fp[1]) for fp in frame_pitches[lo:hi] if fp[1] is not None]
    if names2:
        note2 = Counter(names2).most_common(1)[0][0]
        start = event_samples[i]
        end = event_samples[i + 1] if i + 1 < len(event_samples) else len(y_clean)
        if end - start >= SR // 20:
            seg = y_clean[start:end]
            samples_b2.setdefault(f"sub_{note2}", []).append(seg)
            sequence_b2.append((start / SR, f"sub_{note2}"))

# -----------------------------------------------------------------------
# Create canonical samples (0.4s, fade in/out)
# -----------------------------------------------------------------------
SAMPLE_DUR = 0.4

def make_canonical(segs_dict, prefix):
    canonical = {}
    for name, segs in segs_dict.items():
        best = None
        for seg in segs:
            if len(seg) >= int(SAMPLE_DUR * SR):
                best = seg[:int(SAMPLE_DUR * SR)]
                break
        if best is None:
            best = max(segs, key=lambda s: len(s))
            if len(best) < int(SAMPLE_DUR * SR):
                best = np.pad(best, (0, int(SAMPLE_DUR * SR) - len(best)))
        
        # Fade in/out
        fi = min(int(SR * 0.005), len(best) // 4)
        fo = min(int(SR * 0.03), len(best) // 3)
        if fi > 0:
            best[:fi] *= np.linspace(0, 1, fi)
        if fo > 0:
            best[-fo:] *= np.linspace(1, 0, fo)
        
        safe = name.replace("#", "s").replace("♯", "s").replace("♭", "b")
        key = f"{prefix}_{safe}"
        canonical[key] = best.astype(np.float32)
    return canonical

print("Creating samples...")
canon_b1 = make_canonical(samples_b1, "b1")
canon_b2 = make_canonical(samples_b2, "b2")

# Write WAVs
strudel_map = {}
for name, seg in {**canon_b1, **canon_b2}.items():
    wav_path = SAMPLES_DIR / f"{name}.wav"
    peak = float(np.max(np.abs(seg))) if seg.size else 1.0
    scaled = (seg / max(1e-9, peak) * 32767.0 * 0.9).astype(np.int16)
    wavfile.write(str(wav_path), SR, scaled)
    # Path relative to repo root
    rel = f"Toter%20Schmetterling_stems/bass_samples/{name}.wav"
    strudel_map[name] = rel
    print(f"  {name}.wav")

# Update strudel.json
print("Updating strudel.json...")
STRUDEL_JSON.write_text(json.dumps({"_base": "https://raw.githubusercontent.com/voglll/strudel-converter/main/", **strudel_map}, indent=2))
print(f"  {len(strudel_map)} samples total (bass1: {len(canon_b1)}, bass2: {len(canon_b2)})")

# -----------------------------------------------------------------------
# Generate sample list for the user
# -----------------------------------------------------------------------
print(f"\nBass1 samples ({len(canon_b1)}): {', '.join(sorted(canon_b1.keys()))}")
print(f"Bass2 samples ({len(canon_b2)}): {', '.join(sorted(canon_b2.keys()))}")
