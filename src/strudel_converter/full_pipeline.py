"""
Full automated pipeline: MP3 → Demucs → samples → Strudel

Usage:
  python -m strudel_converter.full_pipeline input.mp3 --output output.txt
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

SR = 22050
HOP = 512
NFFT = 4096


# ---------------------------------------------------------------------------
# Demucs wrapper
# ---------------------------------------------------------------------------

def _run_demucs(mp3_path: Path, output_dir: Path) -> dict[str, Path]:
    """Run Demucs 4-stem separation. Returns {name: wav_path}."""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, 'C:/T'); from demucs import separate; separate.main(['-o', {str(output_dir)!r}, {str(mp3_path)!r}])",
        ],
        check=True, timeout=600,
    )

    # Find output
    model_dir = output_dir / "htdemucs" / mp3_path.stem
    if not model_dir.exists():
        raise FileNotFoundError(f"Demucs output not found: {model_dir}")

    return {
        "drums": model_dir / "drums.wav",
        "bass": model_dir / "bass.wav",
        "other": model_dir / "other.wav",
        "vocals": model_dir / "vocals.wav",
    }


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _hz_to_note(hz: float) -> str:
    if np.isnan(hz) or hz <= 0:
        return "~"
    return librosa.hz_to_note(float(hz), octave=True, unicode=False).replace("#", "s")


def _onsets(y: np.ndarray, hop: int = 256) -> list[int]:
    oe = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    o = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=hop, backtrack=True, units="frames")
    o = sorted(set(o))
    return [x for i, x in enumerate(o) if i == 0 or x - o[i - 1] >= 2]


def _bandpass(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    nyq = SR / 2
    b, a = butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, y)


def _save_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    scaled = (audio / max(1e-9, peak) * 32767 * 0.9).astype(np.int16)
    wavfile.write(str(path), SR, scaled)


# ---------------------------------------------------------------------------
# Drums
# ---------------------------------------------------------------------------

def _process_drums(wav_path: Path, samples_dir: Path, focus_s: float) -> tuple[list[tuple[float, str]], set[str]]:
    """Extract drum samples using band-specific onset detection."""
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y = librosa.util.normalize(y[: int(focus_s * SR)])

    # Band-specific onset detection
    y_kick = _bandpass(y, 30, 200)
    y_snare = _bandpass(y, 400, 6000)
    y_hat = _bandpass(y, 7000, 10000)

    k_ons = set(_onsets(y_kick, 256))
    s_ons = set(_onsets(y_snare, 256))
    h_ons = set(_onsets(y_hat, 256))

    kicks, snares, hats = [], [], []
    seq = []
    used = set()

    for o in sorted(k_ons | s_ons | h_ons):
        if o in used or o < 2:
            continue
        s0 = int(o * 256)
        s1 = min(s0 + int(0.12 * SR), len(y))
        seg = y[s0:s1]
        ik, is_, ih = o in k_ons, o in s_ons, o in h_ons

        if is_ and not ik:
            seg = _bandpass(seg, 200, 5000)
            snares.append(seg)
            label = "dr_snare"
        elif ih and not ik and not is_:
            seg = _bandpass(seg, 6000, 10000)
            hats.append(seg)
            label = "dr_hat"
        elif ik:
            seg = _bandpass(seg, 30, 160)
            kicks.append(seg)
            label = "dr_kick"
        elif is_:
            seg = _bandpass(seg, 200, 5000)
            snares.append(seg)
            label = "dr_snare"
        else:
            seg = _bandpass(seg, 6000, 10000)
            hats.append(seg)
            label = "dr_hat"

        seq.append((o * 256 / SR, label))
        used.add(o)

    # Save canonical samples
    names = set()
    for segs, name in [(kicks, "dr_kick"), (snares, "dr_snare"), (hats, "dr_hat")]:
        if not segs:
            continue
        best = max(segs, key=lambda s: float(np.max(np.abs(s))) if s.size else 0)
        dur = int(0.15 * SR)
        if len(best) >= dur:
            best = best[:dur]
        else:
            best = np.pad(best, (0, dur - len(best)))
        fo = min(int(SR * 0.01), len(best) // 4)
        if fo > 0:
            best[-fo:] *= np.linspace(1, 0, fo)
        _save_wav(samples_dir / f"{name}.wav", best)
        names.add(name)

    return seq, names


# ---------------------------------------------------------------------------
# Bass (harmonic resynthesis)
# ---------------------------------------------------------------------------

def _process_bass(wav_path: Path, samples_dir: Path, focus_s: float) -> tuple[list[tuple[float, str]], set[str]]:
    """Harmonic resynthesis + note segmentation + samples."""
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y = librosa.util.normalize(y[: int(focus_s * SR)])

    # Pitch track
    f0, voiced, _ = librosa.pyin(y, fmin=30, fmax=500, sr=SR, frame_length=NFFT, hop_length=HOP)

    # Harmonic resynthesis
    D_orig = librosa.stft(y, n_fft=NFFT, hop_length=HOP)
    mag_orig = np.abs(D_orig)
    phase_orig = np.angle(D_orig)
    freqs_stft = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    n_frames = min(D_orig.shape[1], len(f0))

    D_synth = np.zeros_like(D_orig, dtype=complex)
    for frame in range(n_frames):
        f = f0[frame]
        is_v = voiced[frame] if frame < len(voiced) else False
        if not is_v or np.isnan(f) or f < 20:
            D_synth[:, frame] = D_orig[:, frame] * 0.25
            continue
        harm_mag = np.zeros(len(freqs_stft))
        for h in range(1, 8):
            hf = f * h
            if hf >= SR / 2:
                break
            idx = np.argmin(np.abs(freqs_stft - hf))
            margin = max(1, int(NFFT * 2 / SR))
            lo = max(0, idx - margin)
            hi = min(len(freqs_stft) - 1, idx + margin)
            he = float(np.max(mag_orig[lo : hi + 1, frame]))
            sigma = max(SR / NFFT * 1.5, hf * 0.01)
            harm_mag += he * np.exp(-0.5 * ((freqs_stft - hf) / sigma) ** 2)
        blended = 0.92 * harm_mag + 0.08 * mag_orig[:, frame]
        oe = np.sum(mag_orig[:, frame]) + 1e-9
        be = np.sum(blended) + 1e-9
        blended *= oe / be
        D_synth[:, frame] = blended * np.exp(1j * phase_orig[:, frame])

    y_synth = librosa.istft(D_synth, hop_length=HOP, length=len(y))
    peak = float(np.max(np.abs(y_synth))) if y_synth.size else 1.0
    if peak > 1e-9:
        y_synth = y_synth / peak * 0.95

    # Pitch on clean synth
    f0_s, _, _ = librosa.pyin(y_synth, fmin=30, fmax=500, sr=SR, frame_length=NFFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y_synth, frame_length=NFFT, hop_length=HOP)[0]
    rms_thresh = float(np.percentile(rms, 15))
    note_names = [_hz_to_note(f) for f in f0_s]

    # Note boundaries
    events = {}
    prev_name = None
    prev_frame = 0
    for i, name in enumerate(note_names):
        if name == "~":
            continue
        if name != prev_name and rms[i] > rms_thresh:
            if (i - prev_frame) * HOP / SR > 0.04:
                events[i] = name
            prev_frame = i
        prev_name = name if name != "~" else prev_name

    # Add onset events
    ons = _onsets(y_synth, HOP)
    for o in ons:
        if o not in events:
            lo, hi = max(0, o - 1), min(len(note_names), o + 4)
            w = [n for n in note_names[lo:hi] if n != "~"]
            if w:
                events[o] = Counter(w).most_common(1)[0][0]

    # Filter and collect segments
    sorted_events = sorted(events.items())
    filtered = [(f, n) for i, (f, n) in enumerate(sorted_events) if i == 0 or f - sorted_events[i - 1][0] >= 2]

    note_segs = {}
    seq = []
    for i, (frame, name) in enumerate(filtered):
        s0 = int(frame * HOP)
        s1 = int(filtered[i + 1][0] * HOP) if i + 1 < len(filtered) else len(y_synth)
        if s1 - s0 < SR // 25:
            continue
        seg = y_synth[s0:s1]
        dur = int(0.5 * SR)
        if len(seg) >= dur:
            seg = seg[:dur]
        else:
            seg = np.pad(seg, (0, dur - len(seg)))
        fi = min(int(SR * 0.005), len(seg) // 6)
        fo = min(int(SR * 0.04), len(seg) // 4)
        if fi > 0:
            seg[:fi] *= np.linspace(0, 1, fi)
        if fo > 0:
            seg[-fo:] *= np.linspace(1, 0, fo)
        key = f"bass_{name}"
        note_segs.setdefault(key, []).append(seg)
        seq.append((s0 / SR, key))

    # Save canonical
    names = set()
    for key, segs in note_segs.items():
        best = max(segs, key=lambda s: float(np.max(np.abs(s))) if s.size else 0)
        _save_wav(samples_dir / f"{key}.wav", best.astype(np.float32))
        names.add(key)

    return seq, names


# ---------------------------------------------------------------------------
# Other (texture/pads)
# ---------------------------------------------------------------------------

def _process_other(wav_path: Path, samples_dir: Path, focus_s: float) -> tuple[list[tuple[float, str]], set[str]]:
    """Simple onset-based segmentation for pads/texture."""
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y = librosa.util.normalize(y[: int(focus_s * SR)])

    f0, _, _ = librosa.pyin(y, fmin=60, fmax=2000, sr=SR, frame_length=NFFT, hop_length=HOP)
    note_names = [_hz_to_note(f) for f in f0]
    ons = _onsets(y, HOP)

    note_segs = {}
    seq = []
    for i, o in enumerate(ons):
        lo, hi = max(0, o - 1), min(len(note_names), o + 4)
        w = [n for n in note_names[lo:hi] if n != "~"]
        if not w:
            continue
        name = Counter(w).most_common(1)[0][0]
        s0 = int(o * HOP)
        s1 = int(ons[i + 1] * HOP) if i + 1 < len(ons) else len(y)
        if s1 - s0 < SR // 20:
            continue
        seg = y[s0:s1]
        dur = int(0.4 * SR)
        if len(seg) >= dur:
            seg = seg[:dur]
        else:
            seg = np.pad(seg, (0, dur - len(seg)))
        fo = min(int(SR * 0.03), len(seg) // 4)
        if fo > 0:
            seg[-fo:] *= np.linspace(1, 0, fo)
        key = f"pad_{name}"
        note_segs.setdefault(key, []).append(seg)
        seq.append((s0 / SR, key))

    names = set()
    for key, segs in note_segs.items():
        best = max(segs, key=lambda s: float(np.max(np.abs(s))) if s.size else 0)
        _save_wav(samples_dir / f"{key}.wav", best.astype(np.float32))
        names.add(key)

    return seq, names


# ---------------------------------------------------------------------------
# Pattern builder
# ---------------------------------------------------------------------------

def _seq_to_pat(seq: list[tuple[float, str]], tempo: float, focus_s: float, spb: int = 4) -> str:
    bd = 60.0 / tempo
    nb = int(focus_s / bd) + 1
    ns = nb * spb
    sd = bd / spb
    grid = [[] for _ in range(ns)]
    for t, k in seq:
        si = int(t / sd)
        if 0 <= si < ns:
            grid[si].append(k)
    bars = []
    for bs in range(0, ns, 16):
        st = []
        for si in range(bs, min(bs + 16, ns)):
            ids = grid[si]
            if not ids:
                st.append("~")
            elif len(ids) == 1:
                st.append(ids[0])
            else:
                st.append("[" + " ".join(ids) + "]")
        bars.append("[" + " ".join(st) + "]")
    return " ".join(bars)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_pipeline(
    mp3_path: str | Path,
    output_txt: str | Path | None = None,
    focus_seconds: float = 15.0,
) -> str:
    mp3_path = Path(mp3_path)
    base = mp3_path.with_suffix("")
    out_dir = base.parent / f"{base.name}_strudel"
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1) Demucs
    print("1/4 Running Demucs 4-stem separation...")
    stem_paths = _run_demucs(mp3_path, out_dir / "demucs")

    # Detect tempo
    y_orig, _ = librosa.load(mp3_path, sr=SR, mono=True)
    y_orig = librosa.util.normalize(y_orig[: int(focus_seconds * SR)])
    tempo, _ = librosa.beat.beat_track(y=y_orig, sr=SR)
    tempo = float(tempo.item() if hasattr(tempo, "item") else tempo)

    # 2) Process each stem
    print(f"2/4 Extracting samples (tempo={tempo:.1f} BPM)...")
    all_seqs = {}
    all_names = {}

    for stem_name, wav_path in stem_paths.items():
        if stem_name == "vocals":
            continue  # skip vocals
        if not wav_path.exists():
            print(f"  {stem_name}: skipped (no file)")
            continue

        if stem_name == "drums":
            seq, names = _process_drums(wav_path, samples_dir, focus_seconds)
        elif stem_name == "bass":
            seq, names = _process_bass(wav_path, samples_dir, focus_seconds)
        else:
            seq, names = _process_other(wav_path, samples_dir, focus_seconds)

        all_seqs[stem_name] = seq
        all_names[stem_name] = names
        print(f"  {stem_name}: {len(seq)} events, {len(names)} samples")

    # 3) Generate strudel.json
    print("3/4 Generating strudel.json...")
    strudel_map = {
        "_base": f"https://raw.githubusercontent.com/voglll/strudel-converter/main/{out_dir.name}/samples/"
    }
    for f in samples_dir.glob("*.wav"):
        strudel_map[f.stem] = str(f.relative_to(out_dir)).replace("\\", "/")
    (out_dir / "strudel.json").write_text(json.dumps(strudel_map, indent=2))

    # 4) Generate Strudel code
    print("4/4 Generating Strudel code...")
    cps = tempo / 60.0
    lines = [
        f"samples('github:voglll/strudel-converter/{out_dir.name}')",
        f"setcps({cps:.4f})",
        "",
    ]

    # Variables
    for stem_name, seq in all_seqs.items():
        pat = _seq_to_pat(seq, tempo, focus_seconds)
        lines.append(f"const {stem_name} = `{pat}`")
    lines.append("")
    lines.append("stack(")

    if "bass" in all_seqs:
        lines.append("  stack(bass).gain(0.9).lpf(500),")
    if "drums" in all_seqs:
        lines.append("  stack(drums).gain(0.85).room(0.05),")
    if "other" in all_seqs:
        lines.append("  stack(other).gain(0.35).room(0.3).lpf(4000),")

    lines.append(")")
    code = "\n".join(lines)

    if output_txt:
        Path(output_txt).write_text(code, encoding="utf-8")
    else:
        (out_dir / "pattern.txt").write_text(code, encoding="utf-8")

    print(f"\nDone. Output: {out_dir}")
    return code
