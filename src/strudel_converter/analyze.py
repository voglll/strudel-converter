from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks

from .models import BeatGrid, Event, PatternTrack, Section, TrackAnalysis


@dataclass(slots=True)
class AnalysisConfig:
    target_sr: int = 22050
    steps_per_bar: int = 64
    loop_bars: int = 2
    beats_per_bar: int = 4
    focus_seconds: float = 5.0
    tempo_min_bpm: float = 90.0
    tempo_max_bpm: float = 190.0
    kick_band_max_hz: int = 160
    snare_band_min_hz: int = 160
    snare_band_max_hz: int = 4500
    hat_band_min_hz: int = 4500
    stft_n_fft: int = 2048
    stft_hop_length: int = 128


def load_audio(path: str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return librosa.util.normalize(y), sr


def _band_energy(magnitude: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    band_mask = (freqs >= low) & (freqs < high)
    if not np.any(band_mask):
        return np.zeros(magnitude.shape[1], dtype=float)
    return magnitude[band_mask].mean(axis=0)


def _estimate_period_from_focus_window(y: np.ndarray, sr: int, config: AnalysisConfig) -> tuple[float, float, np.ndarray]:
    focus_samples = min(len(y), int(config.focus_seconds * sr))
    focus_y = y[:focus_samples]
    if focus_y.size < sr // 2:
        return 120.0, 0.0, focus_y

    hop_length = 256
    onset_env = librosa.onset.onset_strength(y=focus_y, sr=sr, hop_length=hop_length)
    if onset_env.size < 8:
        return 120.0, 0.0, focus_y

    centered = onset_env - np.mean(onset_env)
    window = np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(centered * window))
    freqs = np.fft.rfftfreq(centered.size, d=hop_length / sr)
    bpm_axis = freqs * 60.0

    valid = (bpm_axis >= config.tempo_min_bpm) & (bpm_axis <= config.tempo_max_bpm)
    if not np.any(valid):
        return 120.0, 0.0, focus_y

    valid_spectrum = spectrum[valid]
    valid_bpm = bpm_axis[valid]
    prominence = float(np.percentile(valid_spectrum, 60)) if valid_spectrum.size else 0.0
    peak_indices, _ = find_peaks(valid_spectrum, prominence=prominence)
    if peak_indices.size == 0:
        peak_indices = np.array([int(np.argmax(valid_spectrum))], dtype=int)

    strongest = float(valid_bpm[peak_indices[np.argmax(valid_spectrum[peak_indices])]])
    harmonic_candidates: list[float] = []
    for candidate in (strongest, strongest * 2.0, strongest / 2.0):
        if config.tempo_min_bpm <= candidate <= config.tempo_max_bpm:
            harmonic_candidates.append(float(candidate))
    if not harmonic_candidates:
        harmonic_candidates = [strongest]

    def _comb_score(bpm_value: float) -> tuple[float, float]:
        period_seconds = 60.0 / bpm_value
        period_frames = period_seconds / (hop_length / sr)
        max_phase = max(1, int(round(period_frames)))
        best_score = -np.inf
        best_phase = 0.0
        time_axis = np.arange(onset_env.size, dtype=float)

        for phase in range(max_phase):
            beat_positions = phase + np.arange(0, onset_env.size, period_frames)
            beat_positions = beat_positions[beat_positions < onset_env.size - 1]
            if beat_positions.size < 3:
                continue
            sampled = np.interp(beat_positions, time_axis, onset_env)
            score = float(np.sum(np.maximum(sampled, 0.0)) / (1.0 + np.std(sampled)))
            if score > best_score:
                best_score = score
                best_phase = float(phase)

        if not np.isfinite(best_score):
            best_score = 0.0
        return best_score, best_phase

    scored = [(*_comb_score(bpm), bpm) for bpm in harmonic_candidates]
    _, best_phase_frames, best_bpm = max(scored, key=lambda item: item[0])
    tempo_value = float(best_bpm)
    phase_seconds = float(best_phase_frames * (hop_length / sr))
    return tempo_value, phase_seconds, focus_y


def _step_strength_profiles(
    focus_y: np.ndarray,
    sr: int,
    config: AnalysisConfig,
    tempo_value: float,
    phase_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stft = librosa.stft(focus_y, n_fft=config.stft_n_fft, hop_length=config.stft_hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.stft_n_fft)

    low_band = _band_energy(magnitude, freqs, 20, config.kick_band_max_hz)
    mid_band = _band_energy(magnitude, freqs, config.snare_band_min_hz, config.snare_band_max_hz)
    high_band = _band_energy(magnitude, freqs, config.hat_band_min_hz, sr / 2)

    loop_steps = config.steps_per_bar * config.loop_bars
    low_steps = np.zeros(loop_steps, dtype=float)
    mid_steps = np.zeros(loop_steps, dtype=float)
    high_steps = np.zeros(loop_steps, dtype=float)
    counts = np.zeros(loop_steps, dtype=float)

    bar_duration = (60.0 / tempo_value) * config.beats_per_bar
    loop_duration = bar_duration * config.loop_bars
    frame_times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=config.stft_hop_length)

    for idx, time_value in enumerate(frame_times):
        loop_pos = ((time_value - phase_seconds) % loop_duration) / max(loop_duration, 1e-9)
        loop_step = int(np.clip(np.floor(loop_pos * loop_steps), 0, loop_steps - 1))
        low_steps[loop_step] += float(low_band[idx])
        mid_steps[loop_step] += float(mid_band[idx])
        high_steps[loop_step] += float(high_band[idx])
        counts[loop_step] += 1.0

    counts[counts == 0.0] = 1.0
    low_steps = low_steps / counts
    mid_steps = mid_steps / counts
    high_steps = high_steps / counts

    low_steps = librosa.util.normalize(low_steps)
    mid_steps = librosa.util.normalize(mid_steps)
    high_steps = librosa.util.normalize(high_steps)
    return low_steps, mid_steps, high_steps


def _step_masks_from_profiles(
    low_steps: np.ndarray,
    mid_steps: np.ndarray,
    high_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kick_threshold = float(np.percentile(low_steps, 72))
    snare_threshold = float(np.percentile(mid_steps, 72))
    hat_threshold = float(np.percentile(high_steps, 70))

    kick_mask = (low_steps >= kick_threshold) & (low_steps >= (mid_steps * 1.1))
    snare_mask = (mid_steps >= snare_threshold) & (mid_steps >= (low_steps * 0.95))
    hat_mask = high_steps >= hat_threshold

    if not np.any(kick_mask):
        kick_mask[int(np.argmax(low_steps))] = True
    if not np.any(snare_mask):
        snare_mask[int(np.argmax(mid_steps))] = True
    if not np.any(hat_mask):
        hat_mask[int(np.argmax(high_steps))] = True

    return kick_mask, snare_mask, hat_mask


def _build_event_track(
    name: str,
    step_mask: np.ndarray,
    strength_profile: np.ndarray,
    grid: BeatGrid,
    duration: float,
) -> PatternTrack:
    track = PatternTrack(name=name)
    if not grid.downbeat_times:
        return track

    loop_steps = grid.steps_per_bar * grid.loop_bars
    if step_mask.size != loop_steps:
        return track

    bar_duration = (60.0 / grid.tempo) * grid.beats_per_bar
    first_downbeat = grid.downbeat_times[0]
    peak = float(np.max(strength_profile)) if strength_profile.size else 1.0
    if peak <= 0.0:
        peak = 1.0

    active_steps = np.where(step_mask)[0]
    for bar in range(grid.bars):
        loop_bar = bar % grid.loop_bars
        for loop_step in active_steps:
            if (loop_step // grid.steps_per_bar) != loop_bar:
                continue
            step_in_bar = int(loop_step % grid.steps_per_bar)
            step_center = (step_in_bar + 0.5) / grid.steps_per_bar
            event_time = first_downbeat + bar * bar_duration + step_center * bar_duration
            if event_time > duration:
                continue
            raw_velocity = float(strength_profile[loop_step] / peak)
            velocity = float(np.clip(0.2 + 0.8 * raw_velocity, 0.05, 1.0))
            track.events.append(
                Event(
                    time=event_time,
                    bar=bar,
                    step_in_bar=step_in_bar,
                    velocity=velocity,
                )
            )

    return track


def _pitch_to_name(frequency: float) -> str:
    return librosa.hz_to_note(float(frequency), octave=True, unicode=False).lower()


def _build_bass_track(
    y: np.ndarray,
    sr: int,
    grid: BeatGrid,
    phase_seconds: float,
    duration: float,
) -> PatternTrack:
    track = PatternTrack(name="bass")
    if duration <= 0.0:
        return track

    harmonic, _ = librosa.effects.hpss(y)
    hop_length = 256

    bar_duration = (60.0 / grid.tempo) * grid.beats_per_bar
    loop_duration = bar_duration * grid.loop_bars
    first_downbeat = grid.downbeat_times[0] if grid.downbeat_times else phase_seconds
    loop_start = max(0.0, first_downbeat)
    loop_end = min(duration, loop_start + loop_duration)
    if loop_end <= loop_start:
        return track

    try:
        pyin_pitches, voiced_flag, voiced_prob = librosa.pyin(
            harmonic,
            fmin=librosa.note_to_hz("A1"),
            fmax=librosa.note_to_hz("C5"),
            sr=sr,
            hop_length=hop_length,
        )
    except Exception:
        pyin_pitches = None
        voiced_flag = None
        voiced_prob = None

    if pyin_pitches is not None and voiced_flag is not None and voiced_prob is not None:
        valid_pyin = np.isfinite(pyin_pitches) & voiced_flag & (voiced_prob >= 0.55)
    else:
        valid_pyin = None

    use_pyin = valid_pyin is not None and int(np.sum(valid_pyin)) >= 8

    if not use_pyin:
        yin_pitches = librosa.yin(
            harmonic,
            fmin=librosa.note_to_hz("A1"),
            fmax=librosa.note_to_hz("C5"),
            sr=sr,
            hop_length=hop_length,
        )
        rms = librosa.feature.rms(y=harmonic, frame_length=2048, hop_length=hop_length).flatten()
        rms = np.pad(rms, (0, max(0, len(yin_pitches) - len(rms))), mode="edge")[: len(yin_pitches)]
        rms_threshold = float(np.percentile(rms, 35))
        pitch_track = yin_pitches
        valid_track = np.isfinite(yin_pitches) & (rms >= rms_threshold)
    else:
        pitch_track = pyin_pitches
        valid_track = valid_pyin

    loop_steps = grid.steps_per_bar * grid.loop_bars
    step_edges = np.linspace(loop_start, loop_end, loop_steps + 1)
    notes: list[str] = []

    for start_time, end_time in zip(step_edges[:-1], step_edges[1:]):
        start_frame = int(librosa.time_to_frames(start_time, sr=sr, hop_length=hop_length))
        end_frame = int(librosa.time_to_frames(end_time, sr=sr, hop_length=hop_length))
        end_frame = max(end_frame, start_frame + 1)

        start_frame = int(np.clip(start_frame, 0, len(pitch_track) - 1))
        end_frame = int(np.clip(end_frame, 0, len(pitch_track)))
        if end_frame <= start_frame:
            notes.append("~")
            continue

        frame_slice = pitch_track[start_frame:end_frame]
        valid_slice = valid_track[start_frame:end_frame]
        if frame_slice.size == 0 or not np.any(valid_slice):
            notes.append("~")
            continue

        median_pitch = float(np.median(frame_slice[valid_slice]))
        notes.append(_pitch_to_name(median_pitch))

    if any(note != "~" for note in notes):
        track.notes = notes
    else:
        fallback = ["~"] * loop_steps
        beat_step = max(1, grid.steps_per_bar // grid.beats_per_bar)
        for i in range(0, loop_steps, beat_step):
            fallback[i] = "c1"
        track.notes = fallback

    return track


def _bar_energy_changes(y: np.ndarray, sr: int, grid: BeatGrid, config: AnalysisConfig, duration: float) -> list[Section]:
    if grid.bars <= 1 or not grid.downbeat_times:
        return [Section(0.0, duration, 0, max(grid.bars, 1), "main", 0.5)]

    hop_length = 512
    stft = librosa.stft(y, n_fft=2048, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_band = _band_energy(magnitude, freqs, 20, config.kick_band_max_hz)
    mid_band = _band_energy(magnitude, freqs, config.snare_band_min_hz, config.snare_band_max_hz)
    high_band = _band_energy(magnitude, freqs, config.hat_band_min_hz, sr / 2)
    rms = librosa.feature.rms(S=magnitude).flatten()
    combined = librosa.util.normalize(low_band + mid_band + high_band + rms)

    bar_duration = (60.0 / grid.tempo) * grid.beats_per_bar
    frame_times = librosa.frames_to_time(np.arange(combined.size), sr=sr, hop_length=hop_length)
    first_downbeat = grid.downbeat_times[0]

    bar_values: list[float] = []
    for bar in range(grid.bars):
        start_t = first_downbeat + bar * bar_duration
        end_t = min(duration, start_t + bar_duration)
        if end_t <= start_t:
            bar_values.append(0.0)
            continue
        mask = (frame_times >= start_t) & (frame_times < end_t)
        bar_values.append(float(np.mean(combined[mask])) if np.any(mask) else 0.0)

    if len(bar_values) < 3:
        return [Section(0.0, duration, 0, max(grid.bars, 1), "main", 0.5)]

    deltas = np.abs(np.diff(np.asarray(bar_values, dtype=float)))
    smoothed_deltas = np.convolve(deltas, np.ones(4, dtype=float) / 4.0, mode="same")
    prominence = float(np.percentile(smoothed_deltas, 80)) if smoothed_deltas.size else 0.0
    distance = max(2, grid.bars // 12)
    peaks, _ = find_peaks(smoothed_deltas, prominence=prominence, distance=distance)

    boundaries = sorted({0, *{int(p + 1) for p in peaks}, grid.bars})
    if len(boundaries) <= 2 and grid.bars >= 8:
        boundaries = [0, grid.bars // 2, grid.bars]

    sections: list[Section] = []
    for start_bar, end_bar in zip(boundaries[:-1], boundaries[1:]):
        start_time = max(0.0, first_downbeat + start_bar * bar_duration)
        end_time = min(duration, first_downbeat + end_bar * bar_duration)
        if end_time <= start_time:
            continue
        midpoint = (start_bar + end_bar) / max(grid.bars, 1)
        if midpoint < 0.2:
            label = "intro"
        elif midpoint < 0.55:
            label = "main"
        elif midpoint < 0.8:
            label = "break"
        else:
            label = "outro"
        confidence = float(np.clip(0.4 + 0.03 * (end_bar - start_bar), 0.45, 0.9))
        sections.append(Section(start_time, end_time, start_bar, end_bar, label, confidence))

    return sections or [Section(0.0, duration, 0, max(grid.bars, 1), "main", 0.5)]


def analyze_track(path: str | Path, config: AnalysisConfig | None = None) -> TrackAnalysis:
    config = config or AnalysisConfig()
    y, sr = load_audio(path, config.target_sr)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo_value, phase_seconds, focus_y = _estimate_period_from_focus_window(y, sr, config)

    beat_period = 60.0 / max(tempo_value, 1e-9)
    first_beat = max(0.0, phase_seconds)
    beat_times = list(np.arange(first_beat, duration + beat_period, beat_period))
    if not beat_times:
        beat_times = [0.0]

    downbeat_times = [time for idx, time in enumerate(beat_times) if idx % config.beats_per_bar == 0]
    if not downbeat_times:
        downbeat_times = [first_beat]

    bar_duration = beat_period * config.beats_per_bar
    first_downbeat = downbeat_times[0]
    bars = max(1, int(np.ceil(max(0.0, duration - first_downbeat) / max(bar_duration, 1e-9))))

    grid = BeatGrid(
        tempo=float(tempo_value),
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        bars=bars,
        beats_per_bar=config.beats_per_bar,
        steps_per_bar=config.steps_per_bar,
        loop_bars=config.loop_bars,
    )

    low_steps, mid_steps, high_steps = _step_strength_profiles(
        focus_y=focus_y,
        sr=sr,
        config=config,
        tempo_value=float(tempo_value),
        phase_seconds=phase_seconds,
    )
    kick_mask, snare_mask, hat_mask = _step_masks_from_profiles(low_steps, mid_steps, high_steps)

    tracks = {
        "kick": _build_event_track("kick", kick_mask, low_steps, grid, duration),
        "snare": _build_event_track("snare", snare_mask, mid_steps, grid, duration),
        "hat": _build_event_track("hat", hat_mask, high_steps, grid, duration),
        "bass": _build_bass_track(y, sr, grid, phase_seconds, duration),
    }

    sections = _bar_energy_changes(y, sr, grid, config, duration)
    notes = [
        "Heuristic analysis only; inspect and edit the generated Strudel draft.",
        f"Detected tempo: {tempo_value:.2f} BPM from first {config.focus_seconds:.1f}s Fourier-onset profile.",
        f"Loop profile length: {config.loop_bars} bars at {config.steps_per_bar} steps/bar.",
        "Bass notes extracted from one loop duration with pyin and yin fallback.",
    ]

    return TrackAnalysis(
        path=str(path),
        sample_rate=sr,
        duration=duration,
        grid=grid,
        sections=sections,
        tracks=tracks,
        notes=notes,
    )
