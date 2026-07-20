from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile


def _top_bin_indices(column: np.ndarray, keep: int) -> np.ndarray:
    keep = int(max(1, min(keep, column.size)))
    if keep >= column.size:
        return np.arange(column.size)
    idx = np.argpartition(column, -keep)[-keep:]
    return idx[np.argsort(column[idx])[::-1]]


def _sparse_stft(spectrum: np.ndarray, top_bins_per_frame: int) -> np.ndarray:
    sparse = np.zeros_like(spectrum)
    magnitude = np.abs(spectrum)
    for frame in range(spectrum.shape[1]):
        idx = _top_bin_indices(magnitude[:, frame], top_bins_per_frame)
        sparse[idx, frame] = spectrum[idx, frame]
    return sparse


def _track_partials(
    sparse_spectrum: np.ndarray,
    sample_rate: int,
    n_fft: int,
    partial_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    partial_count = max(1, int(partial_count))
    frame_count = sparse_spectrum.shape[1]
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    partial_freqs = np.full((partial_count, frame_count), np.nan, dtype=float)
    partial_gains = np.zeros((partial_count, frame_count), dtype=float)

    prev_freqs = np.full(partial_count, np.nan, dtype=float)
    magnitude = np.abs(sparse_spectrum)

    for frame in range(frame_count):
        column = magnitude[:, frame]
        peak_bins = _top_bin_indices(column, partial_count)
        peak_freqs = freqs[peak_bins]
        peak_amps = column[peak_bins]

        assigned_peaks = set()
        assigned_partials = set()

        pairs: list[tuple[float, int, int]] = []
        for partial_idx, prev in enumerate(prev_freqs):
            if np.isnan(prev):
                continue
            for peak_idx, freq in enumerate(peak_freqs):
                if peak_amps[peak_idx] <= 0:
                    continue
                dist = abs(float(freq) - float(prev))
                pairs.append((dist, partial_idx, peak_idx))

        pairs.sort(key=lambda item: item[0])
        for _, partial_idx, peak_idx in pairs:
            if partial_idx in assigned_partials or peak_idx in assigned_peaks:
                continue
            partial_freqs[partial_idx, frame] = float(peak_freqs[peak_idx])
            partial_gains[partial_idx, frame] = float(peak_amps[peak_idx])
            assigned_partials.add(partial_idx)
            assigned_peaks.add(peak_idx)

        remaining_partials = [i for i in range(partial_count) if i not in assigned_partials]
        remaining_peaks = [i for i in range(len(peak_bins)) if i not in assigned_peaks and peak_amps[i] > 0]
        for partial_idx, peak_idx in zip(remaining_partials, remaining_peaks, strict=False):
            partial_freqs[partial_idx, frame] = float(peak_freqs[peak_idx])
            partial_gains[partial_idx, frame] = float(peak_amps[peak_idx])

        prev_freqs = partial_freqs[:, frame].copy()

    max_gain = float(np.max(partial_gains)) if partial_gains.size else 1.0
    if max_gain > 0:
        partial_gains = partial_gains / max_gain

    return partial_freqs, partial_gains


def _freq_pattern(values: np.ndarray, gain_values: np.ndarray, silence_threshold: float) -> str:
    tokens: list[str] = []
    for freq, gain in zip(values, gain_values, strict=True):
        if np.isnan(freq) or gain <= silence_threshold:
            tokens.append("~")
        else:
            tokens.append(f"{freq:.2f}")
    return " ".join(tokens)


def _gain_pattern(values: np.ndarray, silence_threshold: float) -> str:
    tokens: list[str] = []
    for gain in values:
        if gain <= silence_threshold:
            tokens.append("0")
        else:
            tokens.append(f"{gain:.3f}")
    return " ".join(tokens)


def _note_name_from_frequency(frequency: float) -> str:
    if not np.isfinite(frequency) or frequency <= 0.0:
        return "~"
    try:
        return librosa.hz_to_note(float(frequency), octave=True, unicode=False).lower()
    except Exception:
        return "~"


def _group_tokens(tokens: list[str], group_size: int = 4) -> str:
    groups: list[str] = []
    for start in range(0, len(tokens), group_size):
        chunk = tokens[start : start + group_size]
        groups.append(f"[{' '.join(chunk)}]")
    return " ".join(groups)


def _bridge_single_gaps(tokens: list[str]) -> list[str]:
    if len(tokens) < 3:
        return tokens
    bridged = tokens[:]
    for index in range(1, len(tokens) - 1):
        if bridged[index] == "~" and bridged[index - 1] == bridged[index + 1] != "~":
            bridged[index] = bridged[index - 1]
    return bridged


def _select_strong_partials(
    partial_freqs: np.ndarray,
    partial_gains: np.ndarray,
    max_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_layers = max(1, int(max_layers))
    mean_gains = np.mean(partial_gains, axis=1) if partial_gains.size else np.array([], dtype=float)
    if mean_gains.size <= max_layers:
        return partial_freqs, partial_gains

    strongest = np.argsort(mean_gains)[::-1][:max_layers]
    strongest = np.sort(strongest)
    return partial_freqs[strongest], partial_gains[strongest]


def _partial_to_note_pattern(
    freq_track: np.ndarray,
    gain_track: np.ndarray,
    target_frames: int,
    silence_threshold: float,
) -> str:
    freq_track = _resample_1d(np.nan_to_num(freq_track, nan=0.0), target_frames=target_frames)
    gain_track = _resample_1d(gain_track, target_frames=target_frames)

    tokens: list[str] = []
    for frequency, gain in zip(freq_track, gain_track, strict=True):
        if gain <= silence_threshold or frequency <= 0.0:
            tokens.append("~")
        else:
            tokens.append(_note_name_from_frequency(float(frequency)))

    tokens = _bridge_single_gaps(tokens)
    return _group_tokens(tokens, group_size=4)


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-12:
        scaled = np.zeros_like(y, dtype=np.int16)
    else:
        normalized = np.clip(y / peak, -1.0, 1.0)
        scaled = (normalized * 32767.0).astype(np.int16)
    wavfile.write(path, sr, scaled)


def _fourier_output_base(output_txt_path: str | Path | None, audio_path: Path) -> Path:
    if output_txt_path is None:
        return audio_path.with_suffix("")
    base_path = Path(output_txt_path)
    stem = base_path.stem
    if stem.endswith("_fourier"):
        stem = stem[: -len("_fourier")]
    return base_path.with_name(stem)


def _resample_partial_tracks(
    partial_freqs: np.ndarray,
    partial_gains: np.ndarray,
    target_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_frames = partial_freqs.shape[1]
    target_frames = int(max(8, target_frames))
    if current_frames <= target_frames:
        return partial_freqs, partial_gains

    x_old = np.linspace(0.0, 1.0, current_frames)
    x_new = np.linspace(0.0, 1.0, target_frames)

    out_freqs = np.full((partial_freqs.shape[0], target_frames), np.nan, dtype=float)
    out_gains = np.zeros((partial_gains.shape[0], target_frames), dtype=float)

    for i in range(partial_freqs.shape[0]):
        gains = partial_gains[i]
        out_gains[i] = np.interp(x_new, x_old, gains)

        valid = np.isfinite(partial_freqs[i]) & (gains > 1e-8)
        if np.count_nonzero(valid) >= 2:
            out_freqs[i] = np.interp(x_new, x_old[valid], partial_freqs[i][valid])

    return out_freqs, out_gains


def _resample_1d(values: np.ndarray, target_frames: int) -> np.ndarray:
    target_frames = int(max(8, target_frames))
    if values.size <= target_frames:
        return values
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, target_frames)
    return np.interp(x_new, x_old, values)


def _estimate_cps(y: np.ndarray, sr: int) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_value = float(np.asarray(tempo).reshape(-1)[0])
        if np.isfinite(tempo_value) and tempo_value > 0.0:
            return tempo_value / 240.0
    except Exception:
        pass
    return 0.65


def fourier_to_strudel(
    audio_path: str | Path,
    output_txt_path: str | Path | None = None,
    analyze_seconds: float = 12.0,
    partial_count: int = 14,
    top_bins_per_frame: int = 56,
    max_frames: int = 128,
    tempo_scale: float = 1.0,
    write_reconstruction_wav: bool = True,
) -> str:
    audio_path = Path(audio_path)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y = librosa.util.normalize(y)

    analyze_seconds = max(2.0, float(analyze_seconds))
    target_len = min(len(y), int(analyze_seconds * sr))
    y = y[:target_len]

    n_fft = 4096
    hop_length = 256
    spectrum = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    sparse_spectrum = _sparse_stft(spectrum, top_bins_per_frame=top_bins_per_frame)

    y_sparse = librosa.istft(sparse_spectrum, hop_length=hop_length, length=len(y), window="hann")
    y_residual = y - y_sparse
    cps = _estimate_cps(y, sr)

    partial_freqs, partial_gains = _track_partials(
        sparse_spectrum,
        sample_rate=sr,
        n_fft=n_fft,
        partial_count=partial_count,
    )

    partial_freqs, partial_gains = _resample_partial_tracks(
        partial_freqs,
        partial_gains,
        target_frames=max_frames,
    )

    partial_freqs, partial_gains = _select_strong_partials(partial_freqs, partial_gains, max_layers=5)

    frame_count = partial_freqs.shape[1]
    silence_threshold = 0.045

    lines: list[str] = []
    lines.append("// Generated by techno-strudel-converter (Fourier experimental mode)")
    lines.append(f"// Source: {audio_path}")
    lines.append("// Model: sparse STFT + sinusoidal partial tracking + residual")
    lines.append(f"// Frames: {frame_count}, Partials: {partial_freqs.shape[0]}")
    lines.append("")
    lines.append("// @by Copilot")
    lines.append("useRNG('legacy')")
    lines.append(f"setcps({cps * max(0.1, float(tempo_scale)):.4f})")
    lines.append("")
    lines.append("stack(")

    rendered: list[str] = []
    for i in range(partial_freqs.shape[0]):
        note_pattern = _partial_to_note_pattern(
            partial_freqs[i],
            partial_gains[i],
            target_frames=frame_count,
            silence_threshold=silence_threshold,
        )
        gain_pattern = _gain_pattern(
            _resample_1d(partial_gains[i], target_frames=frame_count),
            silence_threshold=0.06,
        )
        synth_name = "sine" if i < 3 else "triangle"
        effect_chain = ".gain(.12)" if i < 2 else ".gain(.10)"
        line = (
            f'  // partial {i + 1}\n'
            f'  note("{note_pattern}")'
            f'.s("{synth_name}").gain("{gain_pattern}")'
            f"{effect_chain}.release(.08).jux(rev)"
        )
        rendered.append(line)

    for idx, line in enumerate(rendered):
        suffix = "," if idx < len(rendered) - 1 else ""
        lines.append(f"{line}{suffix}")

    lines.append(")")
    lines.append("")
    lines.append("// Notes")
    lines.append("// - This mode avoids beat-grid/event quantization.")
    lines.append(f"// - Frames were downsampled for Strudel readability (max {max_frames}).")
    lines.append("// - Only the strongest sinusoidal layers are exported to keep the Strudel draft readable.")
    lines.append(f"// - Tempo scale applied to cps: {tempo_scale:.3f}x")
    lines.append("// - Reconstruction is fully spectral and continuous in analysis, then encoded as frame-wise Strudel parameters.")

    if write_reconstruction_wav:
        base_path = _fourier_output_base(output_txt_path, audio_path)
        recon_path = base_path.with_name(base_path.name + "_recon.wav")
        residual_path = base_path.with_name(base_path.name + "_residual.wav")
        _write_wav(recon_path, y_sparse, sr)
        _write_wav(residual_path, y_residual, sr)
        lines.append(f"// Wrote reconstruction WAV: {recon_path}")
        lines.append(f"// Wrote residual WAV: {residual_path}")

    return "\n".join(lines).rstrip() + "\n"
