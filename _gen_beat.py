"""Generate full-beat Strudel stack() pattern from all 4 stems."""
from pathlib import Path
from collections import Counter
import librosa, numpy as np

SR = 22050; NFFT = 4096; HOP = 512; FOCUS = 10.0; TEMPO = 155.88; BD = 60.0/TEMPO
STEMS = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems")
BEAT_SAMPLES = STEMS / "beat_samples"
BASS_SAMPLES = STEMS / "bass_samples"
OUT = Path(r"c:\Users\micha\Desktop\strudel\BASS_LINE.txt")

def hz2name(hz):
    if np.isnan(hz) or hz <= 0: return "~"
    return librosa.hz_to_note(float(hz), octave=True, unicode=False).replace("#", "s")

def get_set(d, prefix):
    return {f.stem for f in d.glob(f"{prefix}*.wav")}

def extract_seq(y, sample_set, prefix):
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP, backtrack=True, units="frames")
    onsets = sorted(set(onsets))
    onsets = [o for i, o in enumerate(onsets) if i == 0 or o - onsets[i - 1] >= 2]

    stft = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    f0, voiced, _ = librosa.pyin(y, fmin=30, fmax=2000, sr=SR, frame_length=NFFT, hop_length=HOP)
    pitch_names = [hz2name(f) for f in f0]
    ke_f = stft[(freqs >= 20) & (freqs <= 160)].mean(axis=0)
    se_f = stft[(freqs >= 160) & (freqs <= 4500)].mean(axis=0)
    he_f = stft[(freqs >= 4500)].mean(axis=0)

    seq = []
    for i, o in enumerate(onsets):
        s0 = int(o * HOP); s1 = int(onsets[i + 1] * HOP) if i + 1 < len(onsets) else len(y)
        if s1 - s0 < SR // 25: continue
        lo, hi = max(0, o - 1), min(len(pitch_names), o + 4)
        if prefix == "dr":
            ke = float(np.mean(ke_f[lo:hi])); se = float(np.mean(se_f[lo:hi])); he = float(np.mean(he_f[lo:hi]))
            mx = max(ke, se, he)
            if mx < 0.001: continue
            label = "dr_kick" if ke == mx else ("dr_snare" if se == mx else "dr_hat")
        else:
            w = [n for n in pitch_names[lo:hi] if n != "~"]
            if not w: continue
            label = f"{prefix}_{Counter(w).most_common(1)[0][0]}"
        if label in sample_set:
            seq.append((s0 / SR, label))
    return seq

def seq_to_pat(seq, steps_per_beat=4):
    nb = int(FOCUS / BD) + 1; ns = nb * steps_per_beat; sd = BD / steps_per_beat
    grid = [[] for _ in range(ns)]
    for t, key in seq:
        si = int(t / sd)
        if 0 <= si < ns: grid[si].append(key)
    bars = []
    for bs in range(0, ns, 16):
        steps = []
        for si in range(bs, min(bs + 16, ns)):
            ids = grid[si]
            if not ids: steps.append("~")
            elif len(ids) == 1: steps.append(ids[0])
            else: steps.append("[" + " ".join(ids) + "]")
        bars.append("[" + " ".join(steps) + "]")
    return " ".join(bars)

# Load
y_b = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_bass.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
y_d = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_drums.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
y_h = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_harmony.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])
y_t = librosa.util.normalize(librosa.load(STEMS / "Toter Schmetterling_texture.wav", sr=SR, mono=True)[0][:int(FOCUS * SR)])

bass_set = get_set(BASS_SAMPLES, "bass_")
drums_set = get_set(BEAT_SAMPLES, "dr_")
harmony_set = get_set(BEAT_SAMPLES, "hm_")
texture_set = get_set(BEAT_SAMPLES, "tx_")

print(f"Bass: {len(bass_set)}  Drums: {len(drums_set)} {drums_set}  Harmony: {len(harmony_set)}  Texture: {len(texture_set)}")

bass_seq = extract_seq(y_b, bass_set, "bass")
drums_seq = extract_seq(y_d, drums_set, "dr")
harmony_seq = extract_seq(y_h, harmony_set, "hm")
texture_seq = extract_seq(y_t, texture_set, "tx")

print(f"Events — bass:{len(bass_seq)} drums:{len(drums_seq)} hm:{len(harmony_seq)} tx:{len(texture_seq)}")

bass_p = seq_to_pat(bass_seq)
drums_p = seq_to_pat(drums_seq)
harmony_p = seq_to_pat(harmony_seq)
texture_p = seq_to_pat(texture_seq)

cps = TEMPO / 60.0
lines = [
    "samples('github:voglll/strudel-converter')",
    f"setcps({cps:.4f})",
    "",
    "stack(",
]
if bass_p.strip(): lines.append(f"  s(`{bass_p}`).gain(0.9).lpf(500),")
if drums_p.strip(): lines.append(f"  s(`{drums_p}`).gain(0.85).room(0.05),")
if harmony_p.strip(): lines.append(f"  s(`{harmony_p}`).gain(0.4).room(0.2).lpf(3000),")
if texture_p.strip(): lines.append(f"  s(`{texture_p}`).gain(0.2).room(0.3).hpf(2000),")
lines.append(")")

code = "\n".join(lines)
OUT.write_text(code, encoding="utf-8")
print(f"\nWrote: {OUT} ({len(code)} bytes)")
print(code)
