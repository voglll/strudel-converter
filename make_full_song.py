"""
Process FULL Demucs stems into bandpassed drum loops (full song length).
Outputs 5 long WAVs + strudel.json + Strudel code.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR_TARGET = 22050  # Keep file sizes manageable (~10MB each)
DEMUCS_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\demucs\htdemucs\Toter Schmetterling")
FULL_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_strudel\full")
CODE_FILE = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_full.txt")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_sr: int = SR_TARGET) -> tuple[np.ndarray, float]:
    """Load WAV, resample to target_sr. Returns (audio, duration_seconds)."""
    import librosa
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / 32768.0
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data, len(data) / target_sr


def save_wav(path: Path, audio: np.ndarray):
    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    if peak > 0:
        audio = audio / peak * 0.95
    wavfile.write(str(path), SR_TARGET, (audio * 32767).astype(np.int16))


def _bandpass(y: np.ndarray, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = SR_TARGET / 2
    sos = butter(order, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Demucs stems (full song)...")
    drums, dur = load_wav(DEMUCS_DIR / "drums.wav")
    bass, _ = load_wav(DEMUCS_DIR / "bass.wav")
    other, _ = load_wav(DEMUCS_DIR / "other.wav")

    print(f"  Duration: {dur:.1f}s ({dur/60:.1f} min)")
    print(f"  Sample rate: {SR_TARGET} Hz")

    # Create output dir
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Process drums into 3 bands (full length) ---
    print("\nSplitting drums into kick/snare/hat (full song)...")
    bands = {
        "kick": (30, 150, 4),
        "snare": (150, 5000, 6),
        "hat": (5000, 11000, 6),
    }

    for name, (lo, hi, order) in bands.items():
        print(f"  {name}: {lo}-{hi} Hz...")
        band = _bandpass(drums, lo, hi, order=order)
        peak = float(np.max(np.abs(band)))
        if peak > 0:
            band = band / peak * 0.95
        save_wav(FULL_DIR / f"{name}_full.wav", band)
        size_mb = (FULL_DIR / f"{name}_full.wav").stat().st_size / 1024 / 1024
        print(f"    -> {name}_full.wav ({size_mb:.1f} MB)")

    # --- Save bass and other (full length, unprocessed) ---
    print("\nSaving bass and other (full song)...")
    save_wav(FULL_DIR / "bass_full.wav", bass)
    save_wav(FULL_DIR / "other_full.wav", other)
    for name in ["bass_full", "other_full"]:
        size_mb = (FULL_DIR / f"{name}.wav").stat().st_size / 1024 / 1024
        print(f"  {name}.wav ({size_mb:.1f} MB)")

    # --- Generate strudel.json ---
    print("\nGenerating strudel.json...")
    gh = "https://raw.githubusercontent.com/voglll/strudel-converter/main/"
    base_path = "Toter%20Schmetterling_strudel"

    strudel_map = {"_base": f"{gh}{base_path}/"}
    for f in FULL_DIR.glob("*.wav"):
        strudel_map[f.stem] = f"full/{f.name}"

    json_path = FULL_DIR.parent / "strudel_full.json"
    json_path.write_text(json.dumps(strudel_map, indent=2))

    # --- Generate Strudel code ---
    total_s = dur
    # One cycle = full song duration
    cps = 1.0 / total_s

    json_url = f"{gh}{base_path}/strudel_full.json"

    code = f"""samples('{json_url}')

// One cycle = entire song ({total_s:.0f}s)
setcps({cps:.6f})

stack(
  s("bass_full").gain(0.95),
  s("other_full").gain(0.5).room(0.15).lpf(6000),
  s("kick_full").gain(0.9),
  s("snare_full").gain(0.85),
  s("hat_full").gain(0.7),
)
"""
    CODE_FILE.write_text(code, encoding="utf-8")

    print(f"\nDone!")
    print(f"  Duration: {total_s:.0f}s ({total_s/60:.1f} min)")
    print(f"  cps: {cps:.6f}")
    print(f"  Code: {CODE_FILE}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
