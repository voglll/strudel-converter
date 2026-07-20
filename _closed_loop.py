"""
Closed-loop drum synthesizer:
  onset detection → synthesize → compare band energy → optimize → WAV samples → Strudel
"""
from pathlib import Path
from collections import Counter
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import librosa, numpy as np, json

SR = 22050; HOP = 256; NFFT = 2048; FOCUS = 10.0; TEMPO = 155.88; BD = 60.0 / TEMPO
ORIG = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling.mp3")
DRUM_STEM = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\Toter Schmetterling_drums.wav")
BASS_STEM = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\Toter Schmetterling_bass.wav")
VERSION = "v2"
BEAT_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\beat_samples")
BASS_DIR = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems\bass_samples")
SJSON = Path(r"c:\Users\micha\Desktop\strudel\strudel.json")
OUT = Path(r"c:\Users\micha\Desktop\strudel\BEAT_v2.txt")

# -----------------------------------------------------------------------
# 1) Onset detection + classification
# -----------------------------------------------------------------------
print("1) Onset detection + classification...")
y_drums, _ = librosa.load(DRUM_STEM, sr=SR, mono=True)
y_drums = librosa.util.normalize(y_drums[:int(FOCUS * SR)])
y_orig, _ = librosa.load(ORIG, sr=SR, mono=True)
y_orig = librosa.util.normalize(y_orig[:int(FOCUS * SR)])

oe = librosa.onset.onset_strength(y=y_drums, sr=SR, hop_length=HOP)
ons = librosa.onset.onset_detect(onset_envelope=oe, sr=SR, hop_length=HOP, backtrack=True, units="frames")
ons = sorted(set(ons))
ons = [o for i, o in enumerate(ons) if i == 0 or o - ons[i - 1] >= 2]

S_stem = np.abs(librosa.stft(y_drums, n_fft=NFFT, hop_length=HOP))
freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
ke = S_stem[(freqs >= 20) & (freqs <= 200)].mean(axis=0)
se = S_stem[(freqs >= 300) & (freqs <= 5000)].mean(axis=0)
he = S_stem[(freqs >= 6000)].mean(axis=0)

S_orig = np.abs(librosa.stft(y_orig, n_fft=NFFT, hop_length=HOP))

kick_ons, snare_ons, hat_ons = [], [], []
for o in ons:
    lo, hi = max(0, o - 1), min(len(ke), o + 3)
    k = float(np.mean(ke[lo:hi]))
    s = float(np.mean(se[lo:hi]))
    h = float(np.mean(he[lo:hi]))
    mx = max(k, s, h)
    if mx < 0.001: continue
    if k == mx: kick_ons.append(o)
    elif s == mx: snare_ons.append(o)
    else: hat_ons.append(o)

print(f"  Kick: {len(kick_ons)}, Snare: {len(snare_ons)}, Hat: {len(hat_ons)}")

# -----------------------------------------------------------------------
# 2) Synthesizer functions
# -----------------------------------------------------------------------
def synth_kick(freq=55.0, bend_ratio=3.0, bend_time=0.015, decay=0.10, vol=1.0):
    """Synthesize a kick drum: pitched sine → exponential decay."""
    dur = int(0.25 * SR)
    t = np.linspace(0, dur / SR, dur, endpoint=False)
    freq_env = freq * (1.0 + bend_ratio * np.exp(-t / bend_time))
    phase = 2.0 * np.pi * np.cumsum(freq_env) / SR
    env = np.exp(-t / decay)
    return (np.sin(phase) * env * vol).astype(np.float32)

def synth_snare(lo_cut=300.0, hi_cut=6000.0, decay=0.06, vol=1.0):
    """Synthesize snare: bandpassed white noise → exponential decay."""
    dur = int(0.15 * SR)
    t = np.linspace(0, dur / SR, dur, endpoint=False)
    noise = np.random.default_rng(42).normal(0, 1, dur).astype(np.float32)
    nyq = SR / 2
    lo = lo_cut / nyq; hi = min(hi_cut, nyq * 0.99) / nyq
    b, a = butter(4, [lo, hi], btype="band")
    filtered = filtfilt(b, a, noise)
    env = np.exp(-t / decay)
    return (filtered * env * vol).astype(np.float32)

def synth_hat(hi_cut=8000.0, decay=0.03, vol=1.0):
    """Synthesize hat: highpassed white noise → fast decay."""
    dur = int(0.08 * SR)
    t = np.linspace(0, dur / SR, dur, endpoint=False)
    noise = np.random.default_rng(99).normal(0, 1, dur).astype(np.float32)
    nyq = SR / 2
    cutoff = min(hi_cut, nyq * 0.95) / nyq
    if cutoff >= 1.0: cutoff = 0.9
    b, a = butter(4, cutoff, btype="highpass")
    filtered = filtfilt(b, a, noise)
    env = np.exp(-t / decay)
    return (filtered * env * vol).astype(np.float32)

# -----------------------------------------------------------------------
# 3) Band energy measurement on original
# -----------------------------------------------------------------------
def band_rms(y, lo_hz, hi_hz):
    if y.size < 64: return 0.0
    n = min(2048, y.size)
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size), n=n))
    f = np.fft.rfftfreq(n, d=1.0 / SR)
    mask = (f >= lo_hz) & (f <= hi_hz)
    return float(np.sqrt(np.mean(spec[mask] ** 2))) if np.any(mask) else 0.0

# Measure per-type average band energy from original
def avg_band_energy(ons_list, lo, hi):
    energies = []
    for o in ons_list:
        s0 = int(o * HOP)
        s1 = min(s0 + int(0.15 * SR), len(y_orig))
        energies.append(band_rms(y_orig[s0:s1], lo, hi))
    return float(np.median(energies)) if energies else 0.0

target_kick_energy = avg_band_energy(kick_ons, 30, 160)
target_snare_energy = avg_band_energy(snare_ons, 300, 5000)
target_hat_energy = avg_band_energy(hat_ons, 6000, 11000)
print(f"\n2) Target band energies:")
print(f"  Kick: {target_kick_energy:.4f} (30-160Hz)")
print(f"  Snare: {target_snare_energy:.4f} (300-5000Hz)")
print(f"  Hat: {target_hat_energy:.4f} (6000-11000Hz)")

# -----------------------------------------------------------------------
# 4) Optimize synth parameters to match band energy
# -----------------------------------------------------------------------
print("\n3) Optimizing synth parameters...")

def optimize(name, synth_fn, param_grid, target_energy, band):
    best_err = float("inf")
    best_params = None
    best_audio = None
    for params in param_grid:
        audio = synth_fn(**params)
        energy = band_rms(audio, *band)
        err = abs(energy - target_energy) / (target_energy + 1e-9)
        if err < best_err:
            best_err = err
            best_params = params
            best_audio = audio
    return best_params, best_audio, best_err

# Kick param grid
kick_grid = []
for freq in [45, 50, 55, 60, 65, 70]:
    for decay in [0.06, 0.08, 0.10, 0.12, 0.15]:
        for vol in [0.5, 0.7, 0.9, 1.0, 1.2]:
            kick_grid.append(dict(freq=freq, decay=decay, vol=vol))

kick_params, kick_audio, kick_err = optimize("kick", synth_kick, kick_grid, target_kick_energy, (30, 160))
print(f"  Kick: freq={kick_params['freq']:.0f}Hz decay={kick_params['decay']:.3f}s vol={kick_params['vol']:.1f} err={kick_err:.3f}")

# Snare param grid
snare_grid = []
for lo_cut in [200, 300, 400, 500]:
    for hi_cut in [3000, 5000, 7000, 9000]:
        for decay in [0.03, 0.05, 0.07, 0.10]:
            for vol in [0.3, 0.5, 0.7, 0.9]:
                snare_grid.append(dict(lo_cut=lo_cut, hi_cut=hi_cut, decay=decay, vol=vol))

snare_params, snare_audio, snare_err = optimize("snare", synth_snare, snare_grid, target_snare_energy, (300, 5000))
print(f"  Snare: lo={snare_params['lo_cut']:.0f}Hz hi={snare_params['hi_cut']:.0f}Hz decay={snare_params['decay']:.3f}s vol={snare_params['vol']:.1f} err={snare_err:.3f}")

# Hat param grid
hat_grid = []
for hi_cut in [5000, 7000, 9000, 10000]:
    for decay in [0.01, 0.02, 0.03, 0.05]:
        for vol in [0.2, 0.3, 0.4, 0.5]:
            hat_grid.append(dict(hi_cut=hi_cut, decay=decay, vol=vol))

if hat_ons:
    hat_params, hat_audio, hat_err = optimize("hat", synth_hat, hat_grid, target_hat_energy, (6000, 11000))
    print(f"  Hat: hi={hat_params['hi_cut']:.0f}Hz decay={hat_params['decay']:.3f}s vol={hat_params['vol']:.1f} err={hat_err:.3f}")
else:
    hat_audio = synth_hat()
    hat_params = {"hi_cut": 8000, "decay": 0.03, "vol": 0.3}
    print(f"  Hat: no onsets — using defaults")

# Clean old unversioned drum WAVs
for old in BEAT_DIR.glob("dr_*.wav"):
    if "_v" not in old.stem:
        old.unlink()

print("\n4) Saving optimized WAV samples...")
def save_wav(audio, name):
    fname = f"{name}_{VERSION}"
    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    scaled = (audio / max(1e-9, peak) * 32767 * 0.9).astype(np.int16)
    wavfile.write(str(BEAT_DIR / f"{fname}.wav"), SR, scaled)
    print(f"  {fname}.wav ({len(audio)/SR*1000:.0f}ms)")
    return fname

k_name = save_wav(kick_audio, "dr_kick")
s_name = save_wav(snare_audio, "dr_snare")
h_name = save_wav(hat_audio, "dr_hat")

# Update strudel.json with versioned names
smap = {"_base": "https://raw.githubusercontent.com/voglll/strudel-converter/main/"}
for f in BEAT_DIR.glob(f"*_{VERSION}.wav"):
    smap[f.stem] = f"Toter%20Schmetterling_stems/beat_samples/{f.name}"
for f in BASS_DIR.glob("bass_*.wav"):
    smap[f.stem] = f"Toter%20Schmetterling_stems/bass_samples/{f.name}"

SJSON.write_text(json.dumps(smap, indent=2))

# -----------------------------------------------------------------------
# 6) Generate Strudel pattern
# -----------------------------------------------------------------------
print("\n5) Generating Strudel pattern...")

def hz2n(hz):
    if np.isnan(hz) or hz <= 0: return "~"
    return librosa.hz_to_note(float(hz), octave=True, unicode=False).replace("#", "s")

# Bass (reuse existing extraction)
yb, _ = librosa.load(BASS_STEM, sr=SR, mono=True)
yb = librosa.util.normalize(yb[:int(FOCUS * SR)])
f0_b, _, _ = librosa.pyin(yb, fmin=30, fmax=500, sr=SR, frame_length=4096, hop_length=HOP)
pn = [hz2n(f) for f in f0_b]
oe_b = librosa.onset.onset_strength(y=yb, sr=SR, hop_length=HOP)
ons_b = sorted(set(librosa.onset.onset_detect(onset_envelope=oe_b, sr=SR, hop_length=HOP, backtrack=True, units="frames")))
ons_b = [o for i, o in enumerate(ons_b) if i == 0 or o - ons_b[i - 1] >= 2]
bset = {f.stem for f in BASS_DIR.glob("bass_*.wav")}
bass_seq = []
for o in ons_b:
    lo, hi = max(0, o - 1), min(len(pn), o + 4)
    w = [n for n in pn[lo:hi] if n != "~"]
    if not w: continue
    note = Counter(w).most_common(1)[0][0]
    key = f"bass_{note}"
    if key in bset: bass_seq.append((o * HOP / SR, key))

# Drum sequence from classified onsets (use versioned names)
drum_seq = []
for o in kick_ons: drum_seq.append((o * HOP / SR, f"dr_kick_{VERSION}"))
for o in snare_ons: drum_seq.append((o * HOP / SR, f"dr_snare_{VERSION}"))
for o in hat_ons: drum_seq.append((o * HOP / SR, f"dr_hat_{VERSION}"))
drum_seq.sort()

def seq_to_pat(seq, spb=4):
    nb = int(FOCUS / BD) + 1; ns = nb * spb; sd = BD / spb
    grid = [[] for _ in range(ns)]
    for t, k in seq:
        si = int(t / sd)
        if 0 <= si < ns: grid[si].append(k)
    bars = []
    for bs in range(0, ns, 16):
        st = []
        for si in range(bs, min(bs + 16, ns)):
            ids = grid[si]
            st.append("~" if not ids else (ids[0] if len(ids) == 1 else "[" + " ".join(ids) + "]"))
        bars.append("[" + " ".join(st) + "]")
    return " ".join(bars)

bp_s = seq_to_pat(bass_seq)
dp_s = seq_to_pat(drum_seq)
cps = TEMPO / 60.0

code = f"samples('github:voglll/strudel-converter')\nsetcps({cps:.4f})\n\nstack(\n  // bass\n  stack(`{bp_s}`).gain(0.9).lpf(500),\n  // drums (synthesized, band-energy-matched)\n  stack(`{dp_s}`).gain(0.85).room(0.05),\n)\n"
OUT.write_text(code)

print(f"  {OUT.name} ({len(code)} bytes)")
print(code[:300] + "...")
print(f"\n✅ BEAT_v2.txt — all samples versioned as *_v2.wav")
