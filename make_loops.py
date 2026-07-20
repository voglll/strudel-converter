"""
Create beat-aligned loops from Demucs stems.
Drums: detect actual pattern, split into kick/snare/hat.
Bass/Other: 2-bar loops.
Samples extracted from FULL song for best quality.
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
LOOP_BARS = 2
DEMUCS_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\demucs\htdemucs\Toter Schmetterling")
SAMPLE_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\samples")
LOOP_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\loops")
STRUDEL_JSON = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\strudel.json")
CODE_FILE = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_full.txt")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_wav(path: Path) -> np.ndarray:
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
    wavfile.write(str(path), SR, (audio * 32767).astype(np.int16))


def _bandpass(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    nyq = SR / 2
    b, a = butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, y)


def _onsets(y: np.ndarray, hop: int = 256) -> list[int]:
    oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    o = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=hop,
                                   backtrack=True, units="frames")
    o = sorted(set(o))
    return [x for i, x in enumerate(o) if i == 0 or x - o[i - 1] >= 2]


# ---------------------------------------------------------------------------
# Beat tracking (on full drums stem)
# ---------------------------------------------------------------------------

def find_loop_start(drums: np.ndarray):
    onset_env = librosa.onset.onset_strength(y=drums, sr=SR)
    tempo, beats = librosa.beat.beat_track(y=drums, sr=SR, onset_envelope=onset_env)
    tempo = float(tempo.item() if hasattr(tempo, 'item') else tempo)

    bar_beats = beats[::4]  # Downbeats (4/4)
    if len(bar_beats) < 2:
        return 0, tempo

    bar_scores = []
    for i, bi in enumerate(bar_beats):
        s0 = librosa.frames_to_samples(bi)
        t = s0 / SR
        lo = max(0, s0 - SR // 8)
        hi = min(len(drums), s0 + SR // 4)
        energy = np.sum(drums[lo:hi] ** 2)
        bar_scores.append((i, energy, t))

    mid_bars = [(i, e, t) for i, e, t in bar_scores if 30 <= t <= 90]
    if len(mid_bars) >= 2:
        best = max(mid_bars[1:], key=lambda x: x[1])
    elif len(bar_scores) >= 3:
        best = max(bar_scores[2:], key=lambda x: x[1])
    else:
        best = max(bar_scores, key=lambda x: x[1])

    start_sample = librosa.frames_to_samples(bar_beats[best[0]])
    print(f"  Tempo: {tempo:.1f} BPM, starting at bar {best[0] + 1}/{len(bar_beats)}")
    return start_sample, tempo


def loop_length_samples(tempo: float, bars: int = LOOP_BARS) -> int:
    beat_dur = 60.0 / tempo
    return int(beat_dur * 4 * bars * SR)


# ---------------------------------------------------------------------------
# Drum splitter: narrow bandpassed loops
# ---------------------------------------------------------------------------

def make_drum_loops(drums_loop: np.ndarray) -> dict:
    """
    Split drums into 3 narrow-band loops:
    - kick:  40-120 Hz   (sub-bass, kick fundamental)
    - snare: 400-3000 Hz  (snare body + crack, avoids guitar mud)
    - hat:   6000-11000 Hz (hi-hats/cymbals, above guitars)
    """
    result = {}
    for name, lo, hi in [("kick", 40, 120), ("snare", 400, 3000), ("hat", 6000, 11000)]:
        band = _bandpass(drums_loop, lo, hi)
        peak = float(np.max(np.abs(band)))
        if peak > 0:
            band = band / peak * 0.95
        result[name] = band
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Demucs stems...")
    drums_full = load_wav(DEMUCS_DIR / "drums.wav")
    bass_full = load_wav(DEMUCS_DIR / "bass.wav")
    other_full = load_wav(DEMUCS_DIR / "other.wav")

    print(f"  drums: {len(drums_full) / SR:.1f}s")
    print(f"  bass:  {len(bass_full) / SR:.1f}s")
    print(f"  other: {len(other_full) / SR:.1f}s")

    # Beat tracking on full drums
    print("\nBeat tracking on drums stem...")
    start_sample, tempo = find_loop_start(drums_full)

    loop_len = loop_length_samples(tempo, LOOP_BARS)
    end_sample = start_sample + loop_len
    loop_dur_s = loop_len / SR
    print(f"  Loop: {LOOP_BARS} bars = {loop_dur_s:.2f}s")

    if end_sample > len(drums_full):
        end_sample = len(drums_full)
        start_sample = max(0, end_sample - loop_len)

    # Extract loop portions
    drums_loop = drums_full[start_sample:end_sample].copy()
    bass_loop = bass_full[start_sample:end_sample].copy()
    other_loop = other_full[start_sample:end_sample].copy()

    # --- Bass & Other: save loops ---
    print("\nSaving bass & other loops...")
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    save_wav(LOOP_DIR / "bass_loop.wav", bass_loop)
    save_wav(LOOP_DIR / "other_loop.wav", other_loop)

    # --- Drums: 3 bandpassed loops ---
    print("\nSplitting drums into kick/snare/hat band-loops...")
    drum_loops = make_drum_loops(drums_loop)
    for name in ['kick', 'snare', 'hat']:
        if name in drum_loops:
            save_wav(LOOP_DIR / f"dr_{name}_loop.wav", drum_loops[name])
            print(f"  dr_{name}_loop.wav: {len(drum_loops[name])/SR:.2f}s")

    # --- Generate strudel.json ---
    print("\nGenerating strudel.json...")
    gh = "https://raw.githubusercontent.com/voglll/strudel-converter/main/"
    base_path = "Toter%20Schmetterling_strudel"

    strudel_map = {"_base": f"{gh}{base_path}/"}
    for f in LOOP_DIR.glob("*.wav"):
        strudel_map[f.stem] = f"loops/{f.name}"

    STRUDEL_JSON.write_text(json.dumps(strudel_map, indent=2))

    # --- Generate Strudel code ---
    cps = tempo / 60.0

    json_filename = "strudel_v5.json"
    STRUDEL_JSON.with_name(json_filename).write_text(json.dumps(strudel_map, indent=2))
    json_url = f"{gh}{base_path}/{json_filename}"

    code = f"""samples('{json_url}')

setcps({cps:.4f})

stack(
  s("bass_loop").loop(1).gain(0.95),
  s("other_loop").loop(1).gain(0.5).room(0.15).lpf(6000),
  s("dr_kick_loop").loop(1).gain(0.9),
  s("dr_snare_loop").loop(1).gain(0.85),
  s("dr_hat_loop").loop(1).gain(0.7),
)
"""
    CODE_FILE.write_text(code, encoding="utf-8")

    print(f"\nDone! Tempo: {tempo:.1f} BPM, {LOOP_BARS}-bar loops")
    print(f"  Loops: bass, other, kick, snare, hat")
    print(f"  Code:  {CODE_FILE}")


if __name__ == "__main__":
    main()
