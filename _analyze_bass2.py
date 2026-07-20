"""
Analyze bass stem for: 1) noise floor, 2) second bass layer.
"""
from pathlib import Path
import librosa
import numpy as np

BASS_WAV = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\Toter Schmetterling_bass.wav")
SR = 22050
N_FFT = 4096
HOP = 512

y, sr = librosa.load(BASS_WAV, sr=SR, mono=True)
y = librosa.util.normalize(y[:int(10 * SR)])  # first 10s

# 1) NOISE FLOOR analysis
print("=== NOISE ANALYSIS ===")
S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

# Energy in bass range (20-160) vs rest
bass_mask = (freqs >= 20) & (freqs <= 160)
noise_mask = (freqs > 200)  # above bass range

bass_energy = float(np.mean(S[bass_mask]))
noise_energy = float(np.mean(S[noise_mask]))
print(f"Bass energy (20-160Hz): {bass_energy:.1f}")
print(f"Noise energy (>200Hz):   {noise_energy:.1f}")
print(f"Signal-to-noise ratio:   {bass_energy/(noise_energy+1e-9):.1f}x")
print(f"Noise floor dB:          {20*np.log10(noise_energy/(bass_energy+1e-9)):.1f} dB")

# Energy per octave
for lo, hi, label in [(20,60,"sub"),(60,120,"low"),(120,250,"mid"),(250,500,"hi-mid"),(500,2000,"presence")]:
    e = float(np.sum(S[(freqs>=lo)&(freqs<hi)]))
    print(f"  {label:10s} {lo:4d}-{hi:4d} Hz: {e:10.1f}")

# 2) MULTI-PITCH detection
print("\n=== MULTI-PITCH ANALYSIS ===")
# For each frame, find the top 2 spectral peaks in bass range
bass_freqs = freqs[(freqs >= 30) & (freqs <= 500)]
bass_S = S[(freqs >= 30) & (freqs <= 500)]

dominant_pitches = []
second_pitches = []
second_ratios = []

for frame in range(S.shape[1]):
    col = bass_S[:, frame]
    if np.max(col) < 0.01 * np.max(S[:, frame]):
        dominant_pitches.append(np.nan)
        second_pitches.append(np.nan)
        second_ratios.append(np.nan)
        continue
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(col, height=np.max(col)*0.05, distance=3)
    if len(peaks) == 0:
        dominant_pitches.append(np.nan)
        second_pitches.append(np.nan)
        second_ratios.append(np.nan)
        continue
    
    # Sort by peak height
    sorted_idx = peaks[np.argsort(col[peaks])[::-1]]
    
    dominant = bass_freqs[sorted_idx[0]]
    dominant_pitches.append(dominant)
    
    if len(sorted_idx) >= 2:
        second = bass_freqs[sorted_idx[1]]
        # Check if second is NOT a harmonic of first
        ratio = second / dominant
        second_pitches.append(second)
        second_ratios.append(ratio)
    else:
        second_pitches.append(np.nan)
        second_ratios.append(np.nan)

dominant_pitches = np.array(dominant_pitches)
second_pitches = np.array(second_pitches)
second_ratios = np.array(second_ratios)

# Analyze second pitch: is it a harmonic or independent?
valid_second = ~np.isnan(second_pitches)
n_valid = np.sum(valid_second)

# Classify ratios: harmonic ratios are near 2.0, 3.0, 4.0, 1.5
harmonic_hits = 0
independent_hits = 0
for r in second_ratios[valid_second]:
    nearest_harmonic = min([1, 1.5, 2, 2.5, 3, 4], key=lambda h: abs(r - h))
    if abs(r - nearest_harmonic) < 0.1:
        harmonic_hits += 1
    else:
        independent_hits += 1

print(f"Frames with second peak: {n_valid}/{len(dominant_pitches)}")
print(f"  Harmonic of dominant:  {harmonic_hits} ({100*harmonic_hits/max(n_valid,1):.0f}%)")
print(f"  Independent pitch:     {independent_hits} ({100*independent_hits/max(n_valid,1):.0f}%)")

if independent_hits > 0:
    indep_pitches = second_pitches[valid_second]
    indep_ratios = second_ratios[valid_second]
    # Filter to independent ones
    is_indep = np.array([min([1,1.5,2,2.5,3,4], key=lambda h: abs(r-h)) for r in indep_ratios])
    is_indep = np.abs(indep_ratios - is_indep) > 0.1
    if np.any(is_indep):
        indep_only = indep_pitches[is_indep]
        print(f"\n  Independent pitch range: {np.min(indep_only):.0f}-{np.max(indep_only):.0f} Hz")
        print(f"  Median independent:      {np.median(indep_only):.0f} Hz ({librosa.hz_to_note(np.median(indep_only))})")
        # Histogram of independent pitches
        from collections import Counter
        notes = [librosa.hz_to_note(p) for p in indep_only]
        note_counts = Counter(notes)
        print(f"  Top independent notes:   {note_counts.most_common(5)}")

# Compare: what octave is dominant vs second?
dom_median = np.nanmedian(dominant_pitches)
sec_median = np.nanmedian(second_pitches)
print(f"\nMedian dominant pitch: {dom_median:.0f} Hz ({librosa.hz_to_note(dom_median)})")
print(f"Median second pitch:   {sec_median:.0f} Hz ({librosa.hz_to_note(sec_median) if not np.isnan(sec_median) else 'N/A'})")
if not np.isnan(sec_median):
    print(f"Frequency ratio:       {sec_median/dom_median:.2f}x")
