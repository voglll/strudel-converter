"""
Closed-loop beat optimizer: classify each beat, synthesize candidate audio,
compare against original, output optimized Strudel patterns.

Pipeline:
  MP3 → beat track → per-beat classifier → mini-synth → compare → optimize → Strudel
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BeatHit:
    """Classification result for a single beat."""

    beat_index: int
    time: float
    kick: bool = False
    snare: bool = False
    hat: bool = False
    kick_energy: float = 0.0
    snare_energy: float = 0.0
    hat_energy: float = 0.0
    bass_note: str = "~"
    bass_hz: float = 0.0
    bass_confidence: float = 0.0


@dataclass(slots=True)
class LoopResult:
    tempo: float
    beat_duration: float
    beats: list[BeatHit] = field(default_factory=list)
    best_error: float = float("inf")
    strudel_code: str = ""


# ---------------------------------------------------------------------------
# per-beat classification
# ---------------------------------------------------------------------------


def _band_rms(y: np.ndarray, sr: int, low: float, high: float) -> float:
    """RMS energy in a frequency band via FFT."""
    if y.size < 16:
        return 0.0
    n = min(2048, y.size)
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size), n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(spec[mask] ** 2)))


def _detect_bass_pitch(y: np.ndarray, sr: int) -> tuple[str, float, float]:
    """Try to detect a bass note in a short audio slice."""
    if y.size < sr // 4:
        return "~", 0.0, 0.0
    # low-pass to isolate bass
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=40, fmax=400, sr=sr, frame_length=2048
        )
    except Exception:
        return "~", 0.0, 0.0

    valid = f0[voiced_flag] if voiced_flag is not None and np.any(voiced_flag) else np.array([])
    if valid.size < 3:
        return "~", 0.0, 0.0

    median_hz = float(np.median(valid))
    conf = float(np.mean(voiced_prob[voiced_flag])) if voiced_prob is not None else 0.5
    try:
        note = librosa.hz_to_note(median_hz, octave=True, unicode=False).lower()
    except Exception:
        note = "~"
    return note, median_hz, conf


def classify_beats(
    y: np.ndarray, sr: int, beat_times: np.ndarray, focus_beats: int = 64
) -> list[BeatHit]:
    """Return per-beat classification for the first *focus_beats* beats."""
    beats: list[BeatHit] = []
    n_beats = min(focus_beats, len(beat_times) - 1)

    for idx in range(n_beats):
        t0 = beat_times[idx]
        t1 = beat_times[idx + 1]
        n0 = int(t0 * sr)
        n1 = int(t1 * sr)
        chunk = y[n0:n1]

        kick_e = _band_rms(chunk, sr, 20, 160)
        snare_e = _band_rms(chunk, sr, 160, 4500)
        hat_e = _band_rms(chunk, sr, 4500, sr // 2)

        # dynamic thresholds: median across all beats
        bass_note, bass_hz, bass_conf = _detect_bass_pitch(chunk, sr)

        beats.append(
            BeatHit(
                beat_index=idx,
                time=float(t0),
                kick_energy=kick_e,
                snare_energy=snare_e,
                hat_energy=hat_e,
                bass_note=bass_note,
                bass_hz=bass_hz,
                bass_confidence=bass_conf,
            )
        )

    # set thresholds from percentiles
    if beats:
        kicks = np.array([b.kick_energy for b in beats])
        snares = np.array([b.snare_energy for b in beats])
        hats = np.array([b.hat_energy for b in beats])

        # use low percentile as "floor" (baseline), higher as "spike"
        k_floor = float(np.percentile(kicks, 20))
        s_floor = float(np.percentile(snares, 20))
        h_floor = float(np.percentile(hats, 20))
        s_spike = float(np.percentile(snares, 65))
        h_spike = float(np.percentile(hats, 65))

        for b in beats:
            # kick: present on nearly every beat in techno (four-on-the-floor)
            b.kick = b.kick_energy >= k_floor * 0.8
            # snare: only when it spikes above baseline AND isn't just kick bleed
            b.snare = b.snare_energy >= s_spike and b.snare_energy > b.kick_energy * 0.08
            # hat: only when it spikes above baseline
            b.hat = b.hat_energy >= h_spike

    return beats


# ---------------------------------------------------------------------------
# mini-synth (Strudel-like sounds in Python)
# ---------------------------------------------------------------------------


def _synth_kick(dur_samples: int, sr: int, freq: float = 55.0, decay: float = 0.08) -> np.ndarray:
    """Sine kick with fast exponential decay."""
    t = np.linspace(0.0, dur_samples / sr, dur_samples, endpoint=False)
    env = np.exp(-t / decay)
    # pitch bend: start higher, drop to fundamental
    freq_env = freq * (1.0 + 2.0 * np.exp(-t / 0.015))
    phase = 2.0 * np.pi * np.cumsum(freq_env) / sr
    return np.sin(phase).astype(np.float32) * env.astype(np.float32)


def _synth_snare(dur_samples: int, sr: int, decay: float = 0.06) -> np.ndarray:
    """Broadband filtered white noise snare — covers 160-4500 Hz."""
    t = np.linspace(0.0, dur_samples / sr, dur_samples, endpoint=False)
    env = np.exp(-t / decay)
    noise = np.random.default_rng(0).normal(0, 1, dur_samples).astype(np.float32)
    # wide bandpass: 200-4000 Hz via FFT
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(dur_samples, d=1.0 / sr)
    # broad bandpass
    lp = 1.0 / (1.0 + (freqs / 3500.0) ** 4)
    hp = 1.0 - 1.0 / (1.0 + (freqs / 250.0) ** 4)
    bp = lp * hp
    filtered = np.fft.irfft(spec * bp.astype(complex), n=dur_samples)
    return (filtered[:dur_samples] * env).astype(np.float32)


def _synth_hat(dur_samples: int, sr: int, decay: float = 0.03) -> np.ndarray:
    """High-frequency noise hat."""
    t = np.linspace(0.0, dur_samples / sr, dur_samples, endpoint=False)
    env = np.exp(-t / decay)
    noise = np.random.default_rng(1).normal(0, 1, dur_samples).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(dur_samples, d=1.0 / sr)
    hp = 1.0 / (1.0 + np.exp(-(freqs - 5000.0) / 800.0))  # sigmoid highpass
    filtered = np.fft.irfft(spec * hp.astype(complex), n=dur_samples)
    return (filtered[:dur_samples] * env * 0.6).astype(np.float32)


def _synth_bass(dur_samples: int, sr: int, freq: float, decay: float = 0.12) -> np.ndarray:
    """Sawtooth bass with lowpass."""
    t = np.linspace(0.0, dur_samples / sr, dur_samples, endpoint=False)
    env = np.exp(-t / decay) * 0.7
    # sawtooth
    phase = np.cumsum(np.full(dur_samples, freq / sr, dtype=np.float32))
    saw = 2.0 * (phase - np.floor(phase)) - 1.0
    # simple lowpass via FFT
    spec = np.fft.rfft(saw * env)
    freqs = np.fft.rfftfreq(dur_samples, d=1.0 / sr)
    lp = 1.0 / (1.0 + (freqs / 500.0) ** 2)  # 2nd order lowpass ~500Hz
    filtered = np.fft.irfft(spec * lp.astype(complex), n=dur_samples)
    return (filtered[:dur_samples] * 0.6).astype(np.float32)


def synthesize_beat(
    hit: BeatHit,
    beat_dur: float,
    sr: int,
    kick_vol: float = 1.0,
    snare_vol: float = 1.8,
    hat_vol: float = 0.8,
    bass_vol: float = 0.5,
) -> np.ndarray:
    """Synthesize one beat from classification.

    Volumes are tuned so that the per-band RMS roughly matches
    what a real techno mix has in each frequency band.
    """
    dur_samples = max(1, int(beat_dur * sr))
    result = np.zeros(dur_samples, dtype=np.float32)

    if hit.kick:
        result += _synth_kick(dur_samples, sr) * kick_vol
    if hit.snare:
        result += _synth_snare(dur_samples, sr) * snare_vol
    if hit.hat:
        result += _synth_hat(dur_samples, sr) * hat_vol
    if hit.bass_hz > 0 and hit.bass_confidence > 0.4:
        result += _synth_bass(dur_samples, sr, hit.bass_hz) * bass_vol

    # gentle peak limit instead of hard normalize (preserves inter-band balance)
    peak = float(np.max(np.abs(result))) if result.size else 1.0
    if peak > 1.0:
        result = result / peak
    return result


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def _spectral_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Weighted spectral distance emphasizing perceptually relevant bands."""
    min_len = min(a.size, b.size)
    if min_len < 64:
        return float(np.mean((a[:min_len] - b[:min_len]) ** 2))

    n_fft = min(2048, min_len)
    spec_a = np.abs(np.fft.rfft(a[:min_len] * np.hanning(min_len), n=n_fft))
    spec_b = np.abs(np.fft.rfft(b[:min_len] * np.hanning(min_len), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # A-weighting approximation: emphasize midrange
    a_weight = 1.0 / (1.0 + (1000.0 / (freqs + 1.0)) ** 2)

    diff = np.abs(spec_a - spec_b) * a_weight
    total_a = np.sum(spec_a * a_weight) + 1e-9
    return float(np.sum(diff) / total_a)


def _banded_spectral_error(
    original: np.ndarray, synth: np.ndarray, sr: int,
    low_hz: float, high_hz: float,
) -> float:
    """Spectral error restricted to a frequency band. Lower = better match."""
    min_len = min(original.size, synth.size)
    if min_len < 64:
        return float(np.mean((original[:min_len] - synth[:min_len]) ** 2))

    n_fft = min(2048, min_len)
    spec_o = np.abs(np.fft.rfft(original[:min_len] * np.hanning(min_len), n=n_fft))
    spec_s = np.abs(np.fft.rfft(synth[:min_len] * np.hanning(min_len), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0

    diff = np.sum(np.abs(spec_o[mask] - spec_s[mask]))
    total_o = np.sum(spec_o[mask]) + 1e-9
    return float(diff / total_o)


def _band_onset(y: np.ndarray, sr: int, low: float, high: float) -> float:
    """Peak onset strength in a frequency band — detects transient hits."""
    if y.size < 256:
        return 0.0
    # bandpass filter
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    lo = max(1.0, low) / nyq
    hi = min(high, nyq * 0.99) / nyq
    if lo >= hi:
        return 0.0
    try:
        b, a = butter(2, [lo, hi], btype="band")
        filtered = filtfilt(b, a, y)
    except Exception:
        return 0.0
    onset = librosa.onset.onset_strength(y=filtered.astype(float), sr=sr)
    # return the peak-to-mean ratio of the onset envelope
    mean_on = float(np.mean(onset)) + 1e-9
    peak_on = float(np.max(onset))
    return peak_on / mean_on


def compare_beat_onset(original: np.ndarray, synth: np.ndarray, sr: int,
                        kick: bool, snare: bool, hat: bool) -> float:
    """Compare onset peak-to-mean ratios per band.

    For each active hit, we check if the synth produces a similar
    transient strength in that band as the original.  For inactive
    hits, we penalize if the synth has stronger transients than
    the original (i.e. we added hits that aren't there).
    """
    bands = [
        ("kick", 20, 160, kick),
        ("snare", 160, 4500, snare),
        ("hat", 4500, sr // 2, hat),
    ]

    total_err = 0.0
    active = 0

    for _name, low, high, is_active in bands:
        o_onset = _band_onset(original, sr, low, high)
        s_onset = _band_onset(synth, sr, low, high)

        if is_active:
            # we want onset strength to be similar
            err = abs(o_onset - s_onset) / (o_onset + 1e-9)
            # cap at 1.0
            err = min(err, 1.0)
            total_err += err
            active += 1
        else:
            # penalty if synth has onset where original doesn't
            if s_onset > o_onset * 1.3 and s_onset > 1.5:
                total_err += min(s_onset / (o_onset + 1e-9) - 1.0, 1.0)

    if active == 0:
        return 1.0
    return total_err / max(active, 1)


# ---------------------------------------------------------------------------
# optimization loop
# ---------------------------------------------------------------------------


def _bit_string(hit: BeatHit) -> str:
    """Encode a hit as a 3-bit string: KSH (kick, snare, hat)."""
    return f"{1 if hit.kick else 0}{1 if hit.snare else 0}{1 if hit.hat else 0}"


def optimize_beats(
    y: np.ndarray,
    sr: int,
    beats: list[BeatHit],
    beat_times: np.ndarray,
    n_beats: int = 16,
) -> tuple[list[BeatHit], float]:
    """Optimize snare/hat decisions per beat; kick is locked to classifier.

    For each beat, try the 4 combos of (snare, hat) while keeping the
    classifier's kick decision fixed.  Pick the combo with lowest
    band-energy error.
    """
    # 4 combos of snare/hat (kick locked to classifier decision)
    combos = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]

    best_beats: list[BeatHit] = []
    total_error = 0.0
    n = min(n_beats, len(beats), len(beat_times) - 1)

    for idx in range(n):
        t0 = beat_times[idx]
        t1 = beat_times[idx + 1]
        original = y[int(t0 * sr) : int(t1 * sr)]
        beat_dur = t1 - t0

        base = beats[idx]
        best_error = float("inf")
        best_hit: BeatHit = base

        for s, h in combos:
            candidate = BeatHit(
                beat_index=idx,
                time=float(t0),
                kick=base.kick,  # locked to classifier
                snare=s,
                hat=h,
                kick_energy=base.kick_energy,
                snare_energy=base.snare_energy,
                hat_energy=base.hat_energy,
                bass_note=base.bass_note,
                bass_hz=base.bass_hz,
                bass_confidence=base.bass_confidence,
            )
            synth = synthesize_beat(candidate, beat_dur, sr)
            err = compare_beat_onset(original, synth, sr, candidate.kick, s, h)
            if err < best_error:
                best_error = err
                best_hit = candidate

        best_beats.append(best_hit)
        total_error += best_error

    return best_beats, total_error / max(n, 1)


# ---------------------------------------------------------------------------
# Strudel code generation
# ---------------------------------------------------------------------------


def _hits_to_strudel(beats: list[BeatHit], tempo: float, steps_per_bar: int = 16) -> str:
    """Convert beat-level hits into a Strudel pattern."""
    beats_per_bar = 4
    total_beats = len(beats)
    bars = max(1, (total_beats + beats_per_bar - 1) // beats_per_bar)
    bars = min(bars, 4)  # cap at 4 bars for readability

    # round to full bars
    total_beats = bars * beats_per_bar

    kick_parts: list[str] = []
    snare_parts: list[str] = []
    hat_parts: list[str] = []
    bass_parts: list[str] = []

    for bar in range(bars):
        bar_k: list[str] = []
        bar_s: list[str] = []
        bar_h: list[str] = []
        bar_b: list[str] = []

        for b in range(beats_per_bar):
            beat_idx = bar * beats_per_bar + b
            if beat_idx < len(beats):
                hit = beats[beat_idx]
                # each beat gets steps_per_bar // beats_per_bar subdivisions
                sub = steps_per_bar // beats_per_bar
                for i in range(sub):
                    if i == 0:
                        bar_k.append("c2" if hit.kick else "~")
                        bar_s.append("white" if hit.snare else "~")
                        bar_h.append("white" if hit.hat else "~")
                        bar_b.append(hit.bass_note if hit.bass_confidence > 0.4 else "~")
                    else:
                        bar_k.append("~")
                        bar_s.append("~")
                        bar_h.append("~")
                        bar_b.append("~")
            else:
                for _ in range(steps_per_bar // beats_per_bar):
                    bar_k.append("~")
                    bar_s.append("~")
                    bar_h.append("~")
                    bar_b.append("~")

        kick_parts.append(f"[{' '.join(bar_k)}]")
        snare_parts.append(f"[{' '.join(bar_s)}]")
        hat_parts.append(f"[{' '.join(bar_h)}]")
        bass_parts.append(f"[{' '.join(bar_b)}]")

    kick_str = " ".join(kick_parts)
    snare_str = " ".join(snare_parts)
    hat_str = " ".join(hat_parts)
    bass_str = " ".join(bass_parts)

    cps = tempo / 60.0

    lines = [
        "// Generated by closed-loop beat optimizer",
        f"// Tempo: {tempo:.2f} BPM  |  {bars} bars × {beats_per_bar} beats",
        "// Per-beat classification + synthesis + spectral comparison loop",
        "",
        f"setcps({cps:.4f})",
        "",
        "stack(",
        f'  note("{kick_str}").sound("sine").decay(0.12).sustain(0).lpf(180).gain(0.95),',
        f'  sound("{snare_str}").decay(0.06).sustain(0).hpf(1500).gain(0.35).room(0.04),',
        f'  sound("{hat_str}").decay(0.03).sustain(0).hpf(8000).gain(0.18).crush(8),',
    ]

    if any(b.bass_confidence > 0.4 for b in beats):
        lines.append(
            f'  note("{bass_str}").sound("sawtooth").decay(0.18).release(0.08).sustain(0.15).lpf(700).gain(0.45).room(0.12),'
        )

    lines.append(")")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run_loop(
    audio_path: str | Path,
    output_txt_path: str | Path | None = None,
    focus_beats: int = 64,
    sr_target: int = 22050,
) -> str:
    """Run the full closed-loop pipeline.

    1. Load audio + beat-track
    2. Per-beat classifier (kick/snare/hat via band energy spikes)
    3. Synthesize audio from classifier decisions
    4. Compare synthesized vs original → quality score
    5. Output Strudel code from classifier + render WAV for listening

    The loop: adjust synth volumes/decay → rerun → lower avg_error = better.
    """
    audio_path = Path(audio_path)

    # 1) load
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    y = librosa.util.normalize(y)

    # 2) beat track
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    tempo_val = float(tempo.item() if hasattr(tempo, "item") else tempo)
    beat_dur = 60.0 / tempo_val

    # 3) classify (this is the primary analysis — no optimizer override)
    beats = classify_beats(y, sr, beat_times, focus_beats=focus_beats)

    # 4) render & compare (quality metric only)
    total_err = 0.0
    rendered = np.zeros(int(min(len(beats), focus_beats) * beat_dur * sr), dtype=np.float32)
    for idx, hit in enumerate(beats):
        t0 = beat_times[idx]
        t1 = beat_times[idx + 1]
        original = y[int(t0 * sr) : int(t1 * sr)]
        synth = synthesize_beat(hit, beat_dur, sr)
        err = compare_beat_onset(original, synth, sr, hit.kick, hit.snare, hit.hat)
        total_err += err

        pos = int(idx * beat_dur * sr)
        end = min(pos + synth.size, rendered.size)
        rendered[pos:end] += synth[: end - pos]

    n = max(len(beats), 1)
    avg_error = total_err / n

    # normalize rendered
    peak = float(np.max(np.abs(rendered))) if rendered.size else 1.0
    if peak > 1e-9:
        rendered = rendered / peak * 0.9

    # 5) strudel code
    code = _hits_to_strudel(beats, tempo_val)

    # write WAV
    out_base = audio_path.with_suffix("")
    if output_txt_path:
        out_base = Path(output_txt_path).with_suffix("")
    recon_path = out_base.parent / (out_base.name + "_loop_recon.wav")
    _write_wav(recon_path, rendered, sr)

    # write report
    report = code + f"\n// --- closed-loop stats ---\n"
    report += f"// Average onset error: {avg_error:.4f}  (lower = better)\n"
    report += f"// Beats analyzed: {len(beats)}\n"
    report += f"// Kick hits: {sum(1 for b in beats if b.kick)}/{len(beats)}\n"
    report += f"// Snare hits: {sum(1 for b in beats if b.snare)}/{len(beats)}\n"
    report += f"// Hat hits: {sum(1 for b in beats if b.hat)}/{len(beats)}\n"
    report += f"// Render: tune --snare-vol / --hat-vol / --kick-decay to reduce error\n"
    report += f"// Reconstruction WAV: {recon_path}\n"

    if output_txt_path:
        Path(output_txt_path).write_text(report, encoding="utf-8")

    return report


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-12:
        scaled = np.zeros_like(y, dtype=np.int16)
    else:
        normalized = np.clip(y / peak, -1.0, 1.0)
        scaled = (normalized * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, scaled)
