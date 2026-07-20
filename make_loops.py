"""
Create properly beat-aligned 2-bar loops from Demucs stems.
Applies noise gate to drums to reduce bleed.
"""

import json
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR = 22050
LOOP_BARS = 2          # 2 bars = more musical context for bass
DEMUCS_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\demucs\htdemucs\Toter Schmetterling")
OUT_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\loops")
STRUDEL_JSON = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\strudel.json")
CODE_FILE = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_full.txt")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_wav(path: Path):
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / 32768.0
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data


def save_wav(path: Path, audio: np.ndarray):
    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    if peak > 0:
        audio = audio / peak * 0.95
    audio_i16 = (audio * 32767).astype(np.int16)
    wavfile.write(str(path), SR, audio_i16)


# ---------------------------------------------------------------------------
# Noise gate for drums
# ---------------------------------------------------------------------------

def noise_gate(audio: np.ndarray, threshold_ratio: float = 0.15, attack_ms: float = 2, release_ms: float = 30) -> np.ndarray:
    """
    Simple RMS noise gate.
    threshold_ratio: fraction of peak RMS below which audio is silenced.
    """
    frame_len = int(SR * attack_ms / 1000)
    hop = frame_len // 4
    release_frames = int(SR * release_ms / 1000) // hop
    
    n_frames = (len(audio) - frame_len) // hop + 1
    rms = np.array([np.sqrt(np.mean(audio[i*hop : i*hop+frame_len]**2))
                    for i in range(n_frames)])
    
    threshold = np.max(rms) * threshold_ratio
    
    env = np.zeros(n_frames)
    for i in range(n_frames):
        if rms[i] > threshold:
            env[i] = 1.0
        elif i > 0 and env[i-1] > 0:
            # Release
            env[i] = max(0, env[i-1] - 1.0 / release_frames)
    
    # Apply envelope
    out = np.zeros_like(audio)
    for i in range(n_frames):
        s0, s1 = i * hop, min(i * hop + frame_len, len(audio))
        out[s0:s1] += audio[s0:s1] * env[i]
    
    # Normalize non-zero parts
    peak_out = float(np.max(np.abs(out)))
    if peak_out > 0:
        out = out / peak_out * 0.95
    
    return out


# ---------------------------------------------------------------------------
# Beat tracking
# ---------------------------------------------------------------------------

def find_loop_start(drums: np.ndarray, tempo: float) -> int:
    """
    Find the best sample index to start a loop.
    Uses onset strength and bar boundaries.
    """
    # Onset strength
    onset_env = librosa.onset.onset_strength(y=drums, sr=SR)
    
    # Beat positions
    tempo, beats = librosa.beat.beat_track(y=drums, sr=SR, onset_envelope=onset_env)
    tempo = float(tempo.item() if hasattr(tempo, 'item') else tempo)
    
    # Bar boundaries (assuming 4/4)
    beats_per_bar = 4
    bar_beats = beats[::beats_per_bar]  # Downbeats
    
    if len(bar_beats) < 2:
        # Fallback: use first beat
        return int(librosa.frames_to_samples(beats[0] if len(beats) > 0 else 0))
    
    # Score each bar start by onset strength on the downbeat
    bar_scores = []
    for i, bi in enumerate(bar_beats):
        s0 = librosa.frames_to_samples(bi)
        t = s0 / SR
        # Look at energy around this downbeat
        lo = max(0, s0 - SR//8)
        hi = min(len(drums), s0 + SR//4)
        energy = np.sum(drums[lo:hi]**2)
        bar_scores.append((i, energy, t))
    
    # Prefer bars in the 30-90 second range (song fully developed)
    # Filter to that range, then pick highest energy
    mid_bars = [(i, e, t) for i, e, t in bar_scores if 30 <= t <= 90]
    if len(mid_bars) >= 2:
        best = max(mid_bars[1:], key=lambda x: x[1])  # skip first in range too
    elif len(bar_scores) >= 3:
        # Fallback: skip first two bars, pick highest energy
        best = max(bar_scores[2:], key=lambda x: x[1])
    else:
        best = max(bar_scores, key=lambda x: x[1])
    
    best_bar_idx = best[0]
    start_sample = librosa.frames_to_samples(bar_beats[best_bar_idx])
    
    print(f"  Tempo: {tempo:.1f} BPM")
    print(f"  Found {len(bar_beats)} bar boundaries, starting at bar {best_bar_idx + 1}/{len(bar_beats)}")
    
    return start_sample, tempo


def loop_length_samples(tempo: float, bars: int = LOOP_BARS) -> int:
    """Samples in `bars` bars at given tempo (4/4)."""
    beat_dur = 60.0 / tempo
    bar_dur = beat_dur * 4
    return int(bar_dur * bars * SR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Demucs stems...")
    drums = load_wav(DEMUCS_DIR / "drums.wav")
    bass = load_wav(DEMUCS_DIR / "bass.wav")
    other = load_wav(DEMUCS_DIR / "other.wav")
    
    print(f"  drums: {len(drums)/SR:.1f}s")
    print(f"  bass:  {len(bass)/SR:.1f}s")
    print(f"  other: {len(other)/SR:.1f}s")
    
    print("\nBeat tracking on drums stem...")
    start_sample, tempo = find_loop_start(drums, tempo=120)
    
    loop_len = loop_length_samples(tempo, LOOP_BARS)
    end_sample = start_sample + loop_len
    
    print(f"  Loop: {LOOP_BARS} bars = {loop_len/SR:.2f}s ({loop_len} samples)")
    print(f"  Start: {start_sample/SR:.2f}s → End: {end_sample/SR:.2f}s")
    
    if end_sample > len(drums):
        print("  WARNING: Loop extends beyond audio, truncating.")
        end_sample = len(drums)
        start_sample = max(0, end_sample - loop_len)
    
    # Extract and save loops
    print("\nExtracting loops...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Drums: apply gate to reduce bleed
    print("  Drums: applying noise gate...")
    drums_loop = drums[start_sample:end_sample].copy()
    drums_loop = noise_gate(drums_loop, threshold_ratio=0.18)
    save_wav(OUT_DIR / "drums_loop.wav", drums_loop)
    
    # Bass: just extract (no processing)
    print("  Bass: extracting...")
    bass_loop = bass[start_sample:end_sample].copy()
    save_wav(OUT_DIR / "bass_loop.wav", bass_loop)
    
    # Other: extract
    print("  Other: extracting...")
    other_loop = other[start_sample:end_sample].copy()
    save_wav(OUT_DIR / "other_loop.wav", other_loop)
    
    # Generate strudel.json
    print("\nGenerating strudel.json...")
    github_base = "https://raw.githubusercontent.com/voglll/strudel-converter/main/"
    strudel_map = {
        "_base": f"{github_base}Toter%20Schmetterling_strudel/loops/",
        "drums": "drums_loop.wav",
        "bass": "bass_loop.wav",
        "other": "other_loop.wav",
    }
    STRUDEL_JSON.write_text(json.dumps(strudel_map, indent=2))
    
    # Generate Strudel code
    cps = tempo / 60.0
    
    # Try different filenames for cache busting
    json_filename = "strudel_v3.json"
    json_url = f"{github_base}Toter%20Schmetterling_strudel/{json_filename}"
    
    # Also write strudel_v3.json to the right place
    STRUDEL_JSON.with_name(json_filename).write_text(json.dumps(strudel_map, indent=2))
    
    code = f"""samples('{json_url}')

setcps({cps:.4f})

stack(
  s("bass").loop(1).gain(0.95),
  s("drums").loop(1).gain(0.85),
  s("other").loop(1).gain(0.5).room(0.15).lpf(6000),
)
"""
    CODE_FILE.write_text(code, encoding="utf-8")
    
    print(f"\nDone! Tempo: {tempo:.1f} BPM, {LOOP_BARS}-bar loops")
    print(f"  Loops: {OUT_DIR}")
    print(f"  Code:  {CODE_FILE}")
    print(f"  JSON:  {STRUDEL_JSON}")


if __name__ == "__main__":
    main()
