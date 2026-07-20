from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


@dataclass(slots=True)
class StemSplitConfig:
    bass_cutoff_hz: float = 180.0
    drum_low_hz: float = 120.0
    drum_high_hz: float = 9000.0
    texture_hp_hz: float = 2500.0
    iterations: int = 4


@dataclass(slots=True)
class StemSplitResult:
    name: str
    samples: np.ndarray
    score: float
    metrics: dict[str, float]


def _normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-12:
        return y.astype(float, copy=True)
    return (y / peak).astype(float, copy=False)


def _design_filter(low_hz: float | None, high_hz: float | None, sr: int, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    nyquist = sr / 2.0
    if low_hz is not None and high_hz is not None:
        low = max(1.0, float(low_hz)) / nyquist
        high = min(float(high_hz), nyquist * 0.99) / nyquist
        low = min(low, 0.99)
        high = max(high, low + 0.01)
        return butter(order, [low, high], btype="band")
    if low_hz is not None:
        cutoff = max(1.0, float(low_hz)) / nyquist
        cutoff = min(cutoff, 0.99)
        return butter(order, cutoff, btype="highpass")
    if high_hz is not None:
        cutoff = max(1.0, float(high_hz)) / nyquist
        cutoff = min(cutoff, 0.99)
        return butter(order, cutoff, btype="lowpass")
    raise ValueError("Either low_hz or high_hz must be provided.")


def _apply_filter(y: np.ndarray, sr: int, low_hz: float | None = None, high_hz: float | None = None) -> np.ndarray:
    if y.size < 32:
        return y.astype(float, copy=True)
    b, a = _design_filter(low_hz=low_hz, high_hz=high_hz, sr=sr)
    try:
        return filtfilt(b, a, y).astype(float, copy=False)
    except Exception:
        return y.astype(float, copy=True)


def _spectral_centroid(y: np.ndarray, sr: int) -> float:
    if y.size < 1024:
        return 0.0
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(centroid)) if centroid.size else 0.0


def _low_band_ratio(y: np.ndarray, sr: int, cutoff_hz: float) -> float:
    if y.size < 1024:
        return 0.0
    spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = float(np.sum(spectrum)) + 1e-9
    low_mask = freqs <= cutoff_hz
    if not np.any(low_mask):
        return 0.0
    low_energy = float(np.sum(spectrum[low_mask]))
    return low_energy / total


def _high_band_ratio(y: np.ndarray, sr: int, cutoff_hz: float) -> float:
    if y.size < 1024:
        return 0.0
    spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = float(np.sum(spectrum)) + 1e-9
    high_mask = freqs >= cutoff_hz
    if not np.any(high_mask):
        return 0.0
    high_energy = float(np.sum(spectrum[high_mask]))
    return high_energy / total


def _drum_transient_score(y: np.ndarray, sr: int) -> float:
    if y.size < 1024:
        return 0.0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0:
        return 0.0
    return float(np.mean(onset_env) / (np.std(onset_env) + 1e-9))


def _candidate_split_from_components(
    y: np.ndarray,
    sr: int,
    config: StemSplitConfig,
    harmonic: np.ndarray,
    percussive: np.ndarray,
) -> dict[str, np.ndarray]:
    bass = _apply_filter(harmonic, sr, high_hz=config.bass_cutoff_hz)
    harmony = harmonic - bass
    drums = _apply_filter(percussive, sr, low_hz=config.drum_low_hz, high_hz=config.drum_high_hz)
    texture = y - bass - harmony - drums
    texture = _apply_filter(texture, sr, low_hz=config.texture_hp_hz)

    return {
        "drums": drums,
        "bass": bass,
        "harmony": harmony,
        "texture": texture,
    }


def _score_split(stems: dict[str, np.ndarray], sr: int, y: np.ndarray, config: StemSplitConfig) -> tuple[float, dict[str, float]]:
    reconstructed = np.zeros_like(y, dtype=float)
    for stem in stems.values():
        reconstructed = reconstructed + stem[: reconstructed.size]
    min_len = min(reconstructed.size, y.size)
    recon_loss = float(np.mean((reconstructed[:min_len] - y[:min_len]) ** 2)) if min_len else 0.0

    bass = stems["bass"]
    drums = stems["drums"]
    harmony = stems["harmony"]
    texture = stems["texture"]

    bass_leak = _high_band_ratio(bass, sr, config.bass_cutoff_hz)
    drum_leak = _low_band_ratio(drums, sr, config.bass_cutoff_hz)
    harmony_leak = _drum_transient_score(harmony, sr)
    texture_energy = float(np.mean(texture ** 2)) if texture.size else 0.0

    score = (
        2.0 * recon_loss
        + 1.4 * bass_leak
        + 1.2 * drum_leak
        + 0.3 * harmony_leak
        + 0.05 * texture_energy
    )

    metrics = {
        "recon_loss": recon_loss,
        "bass_leak": bass_leak,
        "drum_leak": drum_leak,
        "harmony_leak": harmony_leak,
        "texture_energy": texture_energy,
    }
    return score, metrics


def _parameter_grid(base: StemSplitConfig) -> list[StemSplitConfig]:
    bass_cutoffs = [base.bass_cutoff_hz * factor for factor in (0.75, 0.9, 1.0, 1.1, 1.25)]
    drum_lows = [base.drum_low_hz * factor for factor in (0.75, 1.0, 1.2)]
    texture_hps = [base.texture_hp_hz * factor for factor in (0.8, 1.0, 1.2)]
    candidates: list[StemSplitConfig] = []
    for bass_cutoff in bass_cutoffs:
        for drum_low in drum_lows:
            for texture_hp in texture_hps:
                candidates.append(
                    StemSplitConfig(
                        bass_cutoff_hz=float(np.clip(bass_cutoff, 60.0, 500.0)),
                        drum_low_hz=float(np.clip(drum_low, 40.0, 500.0)),
                        drum_high_hz=base.drum_high_hz,
                        texture_hp_hz=float(np.clip(texture_hp, 800.0, 6000.0)),
                        iterations=base.iterations,
                    )
                )
    return candidates


def optimize_stem_split(y: np.ndarray, sr: int, config: StemSplitConfig | None = None) -> tuple[StemSplitConfig, dict[str, np.ndarray], float, dict[str, float]]:
    config = config or StemSplitConfig()
    harmonic, percussive = librosa.effects.hpss(y)
    best_config = config
    best_stems = _candidate_split_from_components(y, sr, best_config, harmonic, percussive)
    best_score, best_metrics = _score_split(best_stems, sr, y, best_config)

    for _ in range(max(1, config.iterations)):
        improved = False
        for candidate in _parameter_grid(best_config):
            stems = _candidate_split_from_components(y, sr, candidate, harmonic, percussive)
            score, metrics = _score_split(stems, sr, y, candidate)
            if score < best_score:
                best_score = score
                best_metrics = metrics
                best_stems = stems
                best_config = candidate
                improved = True
        if not improved:
            break

    return best_config, best_stems, best_score, best_metrics


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-12:
        scaled = np.zeros_like(y, dtype=np.int16)
    else:
        normalized = np.clip(y / peak, -1.0, 1.0)
        scaled = (normalized * 32767.0).astype(np.int16)
    wavfile.write(path, sr, scaled)


def _stem_output_dir(audio_path: Path, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return audio_path.with_name(audio_path.stem + "_stems")


def run_stem_workflow(
    audio_path: str | Path,
    output_txt_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    iterations: int = 4,
) -> str:
    audio_path = Path(audio_path)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y = _normalize_audio(y)

    focus_seconds = 5.0
    focus_samples = min(y.size, int(sr * focus_seconds))
    focus_y = y[:focus_samples]

    config = StemSplitConfig(iterations=iterations)
    best_config, _, score, metrics = optimize_stem_split(focus_y, sr, config)
    full_harmonic, full_percussive = librosa.effects.hpss(y)
    stems = _candidate_split_from_components(y, sr, best_config, full_harmonic, full_percussive)

    stem_dir = _stem_output_dir(audio_path, output_dir)
    stem_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source": str(audio_path),
        "sample_rate": sr,
        "score": score,
        "metrics": metrics,
        "config": {
            "bass_cutoff_hz": best_config.bass_cutoff_hz,
            "drum_low_hz": best_config.drum_low_hz,
            "drum_high_hz": best_config.drum_high_hz,
            "texture_hp_hz": best_config.texture_hp_hz,
            "iterations": best_config.iterations,
        },
        "stems": {},
    }

    stem_paths: dict[str, Path] = {}
    for stem_name, stem_samples in stems.items():
        stem_path = stem_dir / f"{audio_path.stem}_{stem_name}.wav"
        _write_wav(stem_path, stem_samples, sr)
        stem_paths[stem_name] = stem_path
        cast_manifest = manifest["stems"]
        assert isinstance(cast_manifest, dict)
        cast_manifest[stem_name] = {
            "path": str(stem_path),
            "rms": float(np.sqrt(np.mean(stem_samples ** 2))) if stem_samples.size else 0.0,
            "peak": float(np.max(np.abs(stem_samples))) if stem_samples.size else 0.0,
        }

    report_lines: list[str] = []
    report_lines.append("// Generated by techno-strudel-converter (stem-first workflow)")
    report_lines.append(f"// Source: {audio_path}")
    report_lines.append(f"// Optimization window: first {focus_seconds:.0f} seconds")
    report_lines.append(f"// Optimized score: {score:.6f}")
    report_lines.append(f"// Best bass cutoff: {best_config.bass_cutoff_hz:.1f} Hz")
    report_lines.append(f"// Best drum low cutoff: {best_config.drum_low_hz:.1f} Hz")
    report_lines.append(f"// Best texture highpass: {best_config.texture_hp_hz:.1f} Hz")
    report_lines.append("")
    report_lines.append("// Stem files")
    for stem_name, stem_path in stem_paths.items():
        report_lines.append(f"// {stem_name}: {stem_path}")
    report_lines.append(f"// Manifest: {stem_dir / (audio_path.stem + '_stems_manifest.json')}")

    manifest_path = stem_dir / f"{audio_path.stem}_stems_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if output_txt_path is not None:
        Path(output_txt_path).write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    return "\n".join(report_lines).rstrip() + "\n"
