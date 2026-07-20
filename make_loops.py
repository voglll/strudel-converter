"""
Create beat-aligned loops from Demucs stems.
Drums: split into individual kick/snare/hat samples + pattern.
Bass/Other: 2-bar loops.
"""

import json
from collections import Counter
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
# Bandpass filter
# ---------------------------------------------------------------------------

def _bandpass(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    nyq = SR / 2
    b, a = butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, y)


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def _onsets(y: np.ndarray, hop: int = 256) -> list[int]:
    oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    o = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=hop,
                                   backtrack=True, units="frames")
    o = sorted(set(o))
    return [x for i, x in enumerate(o) if i == 0 or x - o[i - 1] >= 2]


# ---------------------------------------------------------------------------
# Beat tracking
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
        lo = max(0, s0 - SR//8)
        hi = min(len(drums), s0 + SR//4)
        energy = np.sum(drums[lo:hi]**2)
        bar_scores.append((i, energy, t))
    
    mid_bars = [(i, e, t) for i, e, t in bar_scores if 30 <= t <= 90]
    if len(mid_bars) >= 2:
        best = max(mid_bars[1:], key=lambda x: x[1])
    elif len(bar_scores) >= 3:
        best = max(bar_scores[2:], key=lambda x: x[1])
    else:
        best = max(bar_scores, key=lambda x: x[1])
    
    start_sample = librosa.frames_to_samples(bar_beats[best[0]])
    print(f"  Tempo: {tempo:.1f} BPM, starting at bar {best[0]+1}/{len(bar_beats)}")
    return start_sample, tempo


def loop_length_samples(tempo: float, bars: int = LOOP_BARS) -> int:
    beat_dur = 60.0 / tempo
    return int(beat_dur * 4 * bars * SR)


# ---------------------------------------------------------------------------
# Drum splitter: kick / snare / hat  (fixed musical pattern)
# ---------------------------------------------------------------------------

def split_drums(drums_loop: np.ndarray, tempo: float, bars: int) -> dict:
    """
    Extract clean kick/snare/hat samples, generate a fixed musical pattern.
    Kick on 1 & 3, snare on 2 & 4, hats on all 8th notes (without kick/snare).
    """
    spb = 4  # 16th notes per beat
    beat_dur = 60.0 / tempo
    slot_dur = beat_dur / spb
    n_slots = bars * 4 * spb  # bars * 4 beats * 4 16ths
    
    # Fixed musical pattern
    pattern = []
    for si in range(n_slots):
        t = si * slot_dur
        beat_in_bar = (si // spb) % 4    # which beat (0-3)
        slot_in_beat = si % spb           # position within beat (0-3)
        
        if slot_in_beat == 0:
            if beat_in_bar in (0, 2):     # Beat 1 or 3 → kick
                pattern.append((t, "kick"))
            elif beat_in_bar in (1, 3):   # Beat 2 or 4 → snare
                pattern.append((t, "snare"))
        elif slot_in_beat == 2:           # 8th note offbeat → hat
            pattern.append((t, "hat"))
    
    # Extract clean samples from actual drum hits in the loop
    y_kick = _bandpass(drums_loop, 30, 200)
    y_snare = _bandpass(drums_loop, 200, 5000)
    y_hat = _bandpass(drums_loop, 5000, 10000)
    
    kick_ons = _onsets(y_kick, 256)
    snare_ons = _onsets(y_snare, 256)
    hat_ons = _onsets(y_hat, 256)
    
    kick_segs, snare_segs, hat_segs = [], [], []
    
    for ons, segs, band_fn in [
        (kick_ons, kick_segs, lambda s: _bandpass(s, 30, 160)),
        (snare_ons, snare_segs, lambda s: _bandpass(s, 200, 5000)),
        (hat_ons, hat_segs, lambda s: _bandpass(s, 5000, 10000)),
    ]:
        for o in ons[:8]:
            s0 = int(o * 256)
            s1 = min(s0 + int(0.15 * SR), len(drums_loop))
            seg = drums_loop[s0:s1].copy()
            seg = band_fn(seg)
            if np.max(np.abs(seg)) > 0.01:
                segs.append(seg)
    
    result = {}
    for segs, name in [(kick_segs, "kick"), (snare_segs, "snare"), (hat_segs, "hat")]:
        if segs:
            best = max(segs, key=lambda s: float(np.max(np.abs(s))) if s.size else 0)
            dur = int(0.15 * SR)
            if len(best) > dur:
                best = best[:dur]
            else:
                best = np.pad(best, (0, dur - len(best)))
            fo = min(int(SR * 0.02), len(best) // 4)
            if fo > 0:
                best[-fo:] *= np.linspace(1, 0, fo)
            result[name] = best
    
    result['pattern'] = pattern
    return result
        # else: silence (no drum hit detected confidently)
    
    # Extract clean samples from the original drums loop
    kick_segs, snare_segs, hat_segs = [], [], []
    
    for t, label in pattern:
        s0 = int(t * SR)
        s1 = min(s0 + int(0.15 * SR), len(drums_loop))
        seg = drums_loop[s0:s1].copy()
        
        if label == "kick":
            seg = _bandpass(seg, 30, 160)
            kick_segs.append(seg)
        elif label == "snare":
            seg = _bandpass(seg, 200, 5000)
            snare_segs.append(seg)
        else:
            seg = _bandpass(seg, 5000, 10000)
            hat_segs.append(seg)
    
    # Pick best samples (loudest, cleanest)
    result = {}
    for segs, name in [(kick_segs, "kick"), (snare_segs, "snare"), (hat_segs, "hat")]:
        if segs:
            best = max(segs, key=lambda s: float(np.max(np.abs(s))) if s.size else 0)
            dur = int(0.15 * SR)
            if len(best) > dur:
                best = best[:dur]
            else:
                best = np.pad(best, (0, dur - len(best)))
            fo = min(int(SR * 0.02), len(best) // 4)
            if fo > 0:
                best[-fo:] *= np.linspace(1, 0, fo)
            result[name] = best
    
    result['pattern'] = pattern
    return result


# ---------------------------------------------------------------------------
# Pattern builder (16th note grid)
# ---------------------------------------------------------------------------

def pattern_to_strudel(pattern: list[tuple[float, str]], tempo: float, loop_dur_s: float) -> str:
    """Convert (time, label) events to Strudel mini-notation."""
    spb = 4  # 16th notes per beat
    beat_dur = 60.0 / tempo
    sd = beat_dur / spb
    ns = int(loop_dur_s / sd)
    grid = [[] for _ in range(ns)]
    
    for t, label in pattern:
        si = int(t / sd)
        if 0 <= si < ns:
            sname = f"dr_{label}"
            if sname not in grid[si]:
                grid[si].append(sname)
    
    # Build mini-notation: split into 16-step bars
    tokens = []
    for si in range(ns):
        ids = grid[si]
        if not ids:
            tokens.append("~")
        elif len(ids) == 1:
            tokens.append(ids[0])
        else:
            tokens.append("[" + " ".join(ids) + "]")
    
    # Group into 16-step bars
    bars = []
    for i in range(0, len(tokens), 16):
        bars.append("[" + " ".join(tokens[i:i+16]) + "]")
    
    return " ".join(bars)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Demucs stems...")
    drums = load_wav(DEMUCS_DIR / "drums.wav")
    bass = load_wav(DEMUCS_DIR / "bass.wav")
    other = load_wav(DEMUCS_DIR / "other.wav")
    
    print(f"  drums: {len(drums)/SR:.1f}s, bass: {len(bass)/SR:.1f}s, other: {len(other)/SR:.1f}s")
    
    print("\nBeat tracking on drums stem...")
    start_sample, tempo = find_loop_start(drums)
    
    loop_len = loop_length_samples(tempo, LOOP_BARS)
    end_sample = start_sample + loop_len
    print(f"  Loop: {LOOP_BARS} bars = {loop_len/SR:.2f}s")
    
    if end_sample > len(drums):
        end_sample = len(drums)
        start_sample = max(0, end_sample - loop_len)
    
    # --- Bass & Other: 2-bar loops ---
    print("\nExtracting bass & other loops...")
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    
    bass_loop = bass[start_sample:end_sample].copy()
    other_loop = other[start_sample:end_sample].copy()
    save_wav(LOOP_DIR / "bass_loop.wav", bass_loop)
    save_wav(LOOP_DIR / "other_loop.wav", other_loop)
    print(f"  bass_loop.wav: {len(bass_loop)/SR:.2f}s")
    print(f"  other_loop.wav: {len(other_loop)/SR:.2f}s")
    
    # --- Drums: split into kick/snare/hat ---
    print("\nSplitting drums into kick / snare / hat...")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    
    drums_loop = drums[start_sample:end_sample].copy()
    drum_data = split_drums(drums_loop, tempo, LOOP_BARS)
    
    drum_samples = {}
    for name in ['kick', 'snare', 'hat']:
        if name in drum_data:
            save_wav(SAMPLE_DIR / f"dr_{name}.wav", drum_data[name])
            drum_samples[name] = f"dr_{name}.wav"
            print(f"  dr_{name}.wav: {len(drum_data[name])/SR*1000:.0f}ms")
    
    drum_pattern = drum_data.get('pattern', [])
    print(f"  Pattern: {len(drum_pattern)} hits")
    
    # --- Generate strudel.json ---
    print("\nGenerating strudel.json...")
    gh = "https://raw.githubusercontent.com/voglll/strudel-converter/main/"
    base_path = "Toter%20Schmetterling_strudel"
    
    strudel_map = {"_base": f"{gh}{base_path}/"}
    for f in SAMPLE_DIR.glob("*.wav"):
        strudel_map[f.stem] = f"samples/{f.name}"
    for f in LOOP_DIR.glob("*.wav"):
        strudel_map[f.stem] = f"loops/{f.name}"
    
    STRUDEL_JSON.write_text(json.dumps(strudel_map, indent=2))
    
    # --- Generate Strudel code ---
    cps = tempo / 60.0
    loop_dur_s = loop_len / SR
    drums_pat = pattern_to_strudel(drum_pattern, tempo, loop_dur_s)
    
    json_filename = "strudel_v4.json"
    STRUDEL_JSON.with_name(json_filename).write_text(json.dumps(strudel_map, indent=2))
    json_url = f"{gh}{base_path}/{json_filename}"
    
    code = f"""samples('{json_url}')

setcps({cps:.4f})

stack(
  s("bass_loop").loop(1).gain(0.95),
  s("other_loop").loop(1).gain(0.5).room(0.15).lpf(6000),
  stack(`{drums_pat}`).gain(0.85),
)
"""
    CODE_FILE.write_text(code, encoding="utf-8")
    
    print(f"\nDone! Tempo: {tempo:.1f} BPM, {LOOP_BARS}-bar loops")
    print(f"  Code:  {CODE_FILE}")
    print(f"\nPreview: {code[:300]}...")


if __name__ == "__main__":
    main()
