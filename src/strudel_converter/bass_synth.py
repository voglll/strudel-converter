"""
Deep bass stem analysis + synthesis.

Pipeline:
  bass.wav → pitch tracking → onset detection → harmonic analysis
           → additive resynthesis → bass_synth.wav → A/B comparison
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def analyze_bass_stem(
    wav_path: str | Path,
    sr_target: int = 22050,
    hop_length: int = 256,
) -> dict:
    """Deep analysis of a bass stem: pitch, onsets, harmonics, envelope."""
    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    y = librosa.util.normalize(y)

    # 1) pitch contour (pyin)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=30, fmax=500, sr=sr,
        frame_length=2048, hop_length=hop_length,
    )
    f0_clean = np.where(voiced_flag, f0, np.nan)

    # 2) RMS envelope per frame
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

    # 3) onset detection on bass
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length,
        backtrack=True, units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # 4) spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=2048, hop_length=hop_length,
    )[0]

    # 5) harmonic strengths (first 8 harmonics relative to fundamental)
    n_fft = 4096
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # for each frame with a voiced pitch, measure harmonic amplitudes
    n_frames = f0_clean.shape[0]
    harmonic_amps = np.zeros((8, n_frames), dtype=float)

    for frame in range(n_frames):
        f = f0_clean[frame]
        if np.isnan(f) or f <= 0:
            continue
        for h in range(1, 9):
            target = f * h
            if target >= sr / 2:
                break
            # find the FFT bin closest to this harmonic
            idx = np.argmin(np.abs(freqs - target))
            # take a small window around the harmonic
            margin = max(1, int(n_fft * 5 / sr))  # ~5 Hz window
            lo = max(0, idx - margin)
            hi = min(len(freqs) - 1, idx + margin)
            harmonic_amps[h - 1, frame] = float(np.max(stft[lo:hi + 1, frame]))

    # normalize harmonic amps per frame
    for frame in range(n_frames):
        total = np.sum(harmonic_amps[:, frame])
        if total > 0:
            harmonic_amps[:, frame] /= total

    result = {
        "audio": y,
        "sr": sr,
        "hop_length": hop_length,
        "f0": f0_clean,
        "f0_raw": f0,
        "voiced_prob": voiced_prob,
        "rms": rms,
        "onset_frames": onset_frames,
        "onset_times": onset_times,
        "centroid": centroid,
        "harmonic_amps": harmonic_amps,
        "freqs": freqs,
        "n_frames": n_frames,
        "duration": len(y) / sr,
    }
    return result


# ---------------------------------------------------------------------------
# synthesis (spectral modeling via STFT/ISTFT — preserves phase)
# ---------------------------------------------------------------------------


def _harmonic_spectrum(
    freq: float,
    harmonic_weights: np.ndarray,
    n_fft: int,
    sr: int,
) -> np.ndarray:
    """Build a synthetic magnitude spectrum for one frame from harmonics."""
    spec = np.zeros(n_fft // 2 + 1, dtype=float)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bin_width = sr / n_fft

    for h_idx, weight in enumerate(harmonic_weights):
        if weight <= 0.001:
            continue
        h_freq = freq * (h_idx + 1)
        if h_freq >= sr / 2:
            break
        # gaussian bump around the harmonic
        sigma = max(bin_width * 2, h_freq * 0.02)  # ~2% bandwidth
        bump = weight * np.exp(-0.5 * ((freqs - h_freq) / sigma) ** 2)
        spec += bump

    return spec


def synthesize_bass_spectral(
    analysis: dict,
    n_fft: int = 2048,
    hop_length: int = 256,
) -> np.ndarray:
    """Spectral modeling synthesis: harmonic model for voiced frames,
    filtered noise for unvoiced, ISTFT reconstruction."""
    y_orig = analysis["audio"]
    sr = analysis["sr"]
    f0 = analysis["f0"]
    rms = analysis["rms"]
    harm = analysis["harmonic_amps"]
    n_frames = analysis["n_frames"]

    # compute STFT of original (we need phase from it for realistic sound)
    D_orig = librosa.stft(y_orig, n_fft=n_fft, hop_length=hop_length)
    mag_orig = np.abs(D_orig)
    phase_orig = np.angle(D_orig)

    # average harmonic profile for unvoiced
    voiced_mask = ~np.isnan(f0)
    if np.any(voiced_mask) and harm.size:
        avg_harm = np.mean(harm[:, voiced_mask], axis=1)
    else:
        avg_harm = np.array([1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.007])

    # build synthetic magnitude spectrum
    D_synth = np.zeros_like(D_orig, dtype=complex)
    n_fft_frames = D_orig.shape[1]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    for frame in range(min(n_fft_frames, n_frames)):
        f = f0[frame]
        frame_rms = rms[frame] if frame < len(rms) else 0.0

        if np.isnan(f) or f <= 20 or frame_rms < 0.001:
            # unvoiced: use filtered noise matching spectral envelope
            # take the original spectrum but smooth it heavily
            if frame < mag_orig.shape[1]:
                # heavily smoothed original spectrum = spectral envelope
                orig_col = mag_orig[:, frame]
                # simple moving average smoothing
                kernel = np.hanning(21)
                kernel = kernel / kernel.sum()
                envelope = np.convolve(orig_col, kernel, mode="same")
                # multiply by random phase
                noise_mag = envelope * 0.3  # quieter for unvoiced
                noise_phase = np.random.default_rng(frame).uniform(0, 2 * np.pi, len(freqs))
                D_synth[:, frame] = noise_mag * np.exp(1j * noise_phase)
        else:
            # voiced: harmonic model
            if frame < harm.shape[1]:
                h_weights = harm[:, frame]
            else:
                h_weights = avg_harm

            harm_spec = _harmonic_spectrum(f, h_weights, n_fft, sr)
            # scale by RMS
            total = np.sum(harm_spec)
            if total > 0:
                harm_spec = harm_spec / total * frame_rms * n_fft * 0.5

            # use original phase for continuity
            if frame < phase_orig.shape[1]:
                phase = phase_orig[:, frame]
            else:
                phase = np.zeros(len(freqs))

            D_synth[:, frame] = harm_spec * np.exp(1j * phase)

    # ISTFT
    y_synth = librosa.istft(D_synth, hop_length=hop_length, length=len(y_orig))

    # normalize
    peak = float(np.max(np.abs(y_synth))) if y_synth.size else 1.0
    if peak > 1e-9:
        y_synth = y_synth / peak * 0.95

    return y_synth


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_bass_pipeline(
    bass_wav: str | Path,
    output_wav: str | Path | None = None,
) -> dict:
    """Full bass analysis → spectral synthesis → comparison pipeline."""
    bass_wav = Path(bass_wav)

    print(f"Analyzing: {bass_wav}")
    analysis = analyze_bass_stem(bass_wav)

    dur = analysis["duration"]
    voiced_pct = np.mean(~np.isnan(analysis["f0"])) * 100
    n_onsets = len(analysis["onset_times"])
    median_pitch = float(np.nanmedian(analysis["f0"]))

    print(f"  Duration: {dur:.1f}s")
    print(f"  Voiced frames: {voiced_pct:.1f}%")
    print(f"  Onsets detected: {n_onsets}")
    print(f"  Median pitch: {median_pitch:.1f} Hz "
          f"({librosa.hz_to_note(median_pitch) if median_pitch > 0 else '?'})")

    # average harmonic profile
    harm = analysis["harmonic_amps"]
    voiced = ~np.isnan(analysis["f0"])
    if np.any(voiced) and harm.size:
        avg = np.mean(harm[:, voiced], axis=1)
        print(f"  Avg harmonics: "
              f"{' '.join(f'h{i+1}={avg[i]:.3f}' for i in range(min(6, len(avg))))}")

    print("Synthesizing (spectral modeling)...")
    synthesized = synthesize_bass_spectral(analysis)

    if output_wav is None:
        output_wav = bass_wav.with_name(bass_wav.stem + "_synth.wav")

    # write synthesized
    peak = float(np.max(np.abs(synthesized))) if synthesized.size else 1.0
    if peak > 1e-9:
        scaled = (synthesized / peak * 32767.0).astype(np.int16)
    else:
        scaled = np.zeros_like(synthesized, dtype=np.int16)
    wavfile.write(str(output_wav), analysis["sr"], scaled)
    print(f"  Wrote: {output_wav}")

    # A/B comparison
    orig = analysis["audio"]
    synth = synthesized
    min_len = min(orig.size, synth.size)

    # spectral error
    n_fft_cmp = 2048
    hop_cmp = 512
    S_orig = np.abs(librosa.stft(orig[:min_len], n_fft=n_fft_cmp, hop_length=hop_cmp))
    S_synth = np.abs(librosa.stft(synth[:min_len], n_fft=n_fft_cmp, hop_length=hop_cmp))
    spectral_err = float(np.mean(np.abs(S_orig - S_synth)) / (np.mean(S_orig) + 1e-9))

    # correlation (first 30s to avoid memory issues)
    compare_len = min(min_len, 30 * analysis["sr"])
    corr = float(np.corrcoef(orig[:compare_len], synth[:compare_len])[0, 1])

    print(f"  Spectral error: {spectral_err:.4f}  (lower = better)")
    print(f"  Waveform correlation: {corr:.4f}  (higher = better)")

    return {
        "analysis": analysis,
        "synthesized": synthesized,
        "spectral_error": spectral_err,
        "correlation": corr,
        "output_wav": output_wav,
    }
