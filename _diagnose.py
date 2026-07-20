"""
Diagnostic plot: understand the drum stem and why snare extraction fails.
Shows waveform, spectrogram, onset positions, and per-hit frequency analysis.
"""
from pathlib import Path
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa, numpy as np

SR = 22050
ORIG = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling.mp3")
DRUMS = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\Toter Schmetterling_drums.wav")

y_orig, _ = librosa.load(ORIG, sr=SR, mono=True)
y_orig = librosa.util.normalize(y_orig[:int(8 * SR)])
y_drums, _ = librosa.load(DRUMS, sr=SR, mono=True)
y_drums = librosa.util.normalize(y_drums[:int(8 * SR)])

HOP = 256
NFFT = 2048

# Beat track
tempo, beats = librosa.beat.beat_track(y=y_orig, sr=SR, units="frames")
beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
tempo_val = float(tempo.item())

# Onsets on drum stem
oe = librosa.onset.onset_strength(y=y_drums, sr=SR, hop_length=HOP)
ons = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=HOP, backtrack=True, units="frames")

# Spectrograms
S_drums = librosa.amplitude_to_db(np.abs(librosa.stft(y_drums, n_fft=NFFT, hop_length=HOP)), ref=np.max)
S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig, n_fft=NFFT, hop_length=HOP)), ref=np.max)

# Band energies per onset
stft = np.abs(librosa.stft(y_drums, n_fft=NFFT, hop_length=HOP))
freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
ke = stft[(freqs >= 20) & (freqs <= 200)].mean(axis=0)
se = stft[(freqs >= 300) & (freqs <= 5000)].mean(axis=0)
he = stft[(freqs >= 6000)].mean(axis=0)

fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# 1) Original vs Drum stem waveform (first 4 beats)
ax = axes[0]
dur = min(4 * 60.0 / tempo_val, 8)
n = int(dur * SR)
t = np.linspace(0, dur, n)
ax.plot(t, y_orig[:n], "steelblue", lw=0.4, alpha=0.7, label="Original")
ax.plot(t, y_drums[:n], "darkorange", lw=0.6, label="Drum Stem")
for bt in beat_times[:6]:
    ax.axvline(bt, color="cyan", ls="--", lw=1, alpha=0.5)
ax.legend(fontsize=8)
ax.set_title(f"Waveform — first {dur:.0f}s (cyan = beat boundaries, {tempo_val:.0f} BPM)")
ax.set_ylabel("Amplitude")
ax.grid(alpha=0.3)

# 2) Drum stem spectrogram with onsets
ax = axes[1]
librosa.display.specshow(S_drums, sr=SR, hop_length=HOP, x_axis="s", y_axis="hz", ax=ax, cmap="magma")
ax.set_ylim(0, 8000)
# Mark onsets
for o in ons:
    t_o = o * HOP / SR
    ax.axvline(t_o, color="lime", ls="-", lw=0.8, alpha=0.6)
# Mark beat boundaries
for bt in beat_times[:10]:
    ax.axvline(bt, color="cyan", ls="--", lw=1.2, alpha=0.7)
ax.set_title(f"Drum Stem Spectrogram (green=onsets, cyan=beats)")

# 3) Band energies + classification
ax = axes[2]
frame_t = librosa.frames_to_time(np.arange(len(ke)), sr=SR, hop_length=HOP)
ax.plot(frame_t, ke / (np.max(ke) + 1e-9), "red", lw=0.7, alpha=0.8, label="Kick band (20-200Hz)")
ax.plot(frame_t, se / (np.max(se) + 1e-9), "green", lw=0.7, alpha=0.8, label="Snare band (300-5000Hz)")
ax.plot(frame_t, he / (np.max(he) + 1e-9), "blue", lw=0.5, alpha=0.6, label="Hat band (6000+Hz)")
# Classify each onset
for o in ons:
    lo, hi = max(0, o - 1), min(len(ke), o + 3)
    k, s, h = float(np.mean(ke[lo:hi])), float(np.mean(se[lo:hi])), float(np.mean(he[lo:hi]))
    mx = max(k, s, h)
    color = "red" if k == mx else ("green" if s == mx else "blue")
    label = "K" if k == mx else ("S" if s == mx else "H")
    t_o = o * HOP / SR
    ax.axvline(t_o, color=color, ls=":", lw=1, alpha=0.5)
    ax.text(t_o, 1.05, label, color=color, fontsize=7, ha="center")
ax.legend(fontsize=7)
ax.set_title(f"Band Energies + Classification (K=kick, S=snare, H=hat)")
ax.set_ylabel("Normalized energy")
ax.grid(alpha=0.3)
ax.set_xlim(0, dur)

# 4) Zoom: first 2 beats — waveform + onset regions
ax = axes[3]
zoom_dur = min(2 * 60.0 / tempo_val, 3)
zn = int(zoom_dur * SR)
zt = np.linspace(0, zoom_dur, zn)
ax.plot(zt, y_drums[:zn], "darkorange", lw=0.6)
# Highlight onset regions
for o in ons:
    t_o = o * HOP / SR
    if t_o > zoom_dur: break
    lo, hi = max(0, o - 1), min(len(ke), o + 3)
    k, s, h = float(np.mean(ke[lo:hi])), float(np.mean(se[lo:hi])), float(np.mean(he[lo:hi]))
    mx = max(k, s, h)
    color = "red" if k == mx else ("green" if s == mx else "blue")
    label = "KICK" if k == mx else ("SNARE" if s == mx else "HAT")
    # Shade the extracted region
    s0 = int(o * HOP)
    s1 = min(s0 + int(0.06 * SR), zn) if s == mx else min(s0 + int(0.1 * SR), zn)
    ax.axvspan(s0 / SR, s1 / SR, alpha=0.15, color=color)
    ax.text(t_o, ax.get_ylim()[1] * 0.9, label, color=color, fontsize=6, ha="center", rotation=90)
ax.set_title("Drum Stem — Onset Regions (red=kick, green=snare, blue=hat)")
ax.set_xlabel("Time [s]")
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path(r"c:\Users\micha\Desktop\strudel\_drum_diagnostic.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# Print stats
n_k = sum(1 for o in ons if float(np.mean(ke[max(0,o-1):min(len(ke),o+3)])) >= max(float(np.mean(se[max(0,o-1):min(len(se),o+3)])), float(np.mean(he[max(0,o-1):min(len(he),o+3)]))))
n_s = sum(1 for o in ons if float(np.mean(se[max(0,o-1):min(len(se),o+3)])) > float(np.mean(ke[max(0,o-1):min(len(ke),o+3)])) and float(np.mean(se[max(0,o-1):min(len(se),o+3)])) >= float(np.mean(he[max(0,o-1):min(len(he),o+3)])))
print(f"Total onsets: {len(ons)}, Kick: {n_k}, Snare: {n_s}")
# Show snare-band to kick-band ratio for first 10 onsets
print("\nFirst 15 onsets — k/s ratio:")
for o in ons[:15]:
    lo, hi = max(0, o - 1), min(len(ke), o + 3)
    k, s = float(np.mean(ke[lo:hi])), float(np.mean(se[lo:hi]))
    print(f"  {o*HOP/SR:.2f}s: kick={k:.4f} snare={s:.4f} ratio={s/(k+1e-9):.2f} → {'SNARE' if s>k else 'KICK'}")
