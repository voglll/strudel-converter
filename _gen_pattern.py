"""Generate BASS_LINE.txt from drum stem + bass stem."""
from pathlib import Path
from collections import Counter
import librosa, numpy as np

SR=22050; FOCUS=10.0; TEMPO=155.88; BD=60.0/TEMPO
STEMS = Path(r"c:\Users\micha\Desktop\strudel\Toter Schmetterling_stems")
BEAT = STEMS/"beat_samples"; BASS=STEMS/"bass_samples"
OUT = Path(r"c:\Users\micha\Desktop\strudel\BASS_LINE.txt")

def hz2n(hz):
    if np.isnan(hz) or hz<=0: return "~"
    return librosa.hz_to_note(float(hz),octave=True,unicode=False).replace("#","s")

def onsets(y):
    oe = librosa.onset.onset_strength(y=y,sr=SR,hop_length=256)
    o = librosa.onset.onset_detect(onset_envelope=oe,sr=SR,hop_length=256,backtrack=True,units="frames")
    return sorted(set(o))

yd = librosa.util.normalize(librosa.load(STEMS/"Toter Schmetterling_drums.wav",sr=SR,mono=True)[0][:int(FOCUS*SR)])
d_ons = onsets(yd); d_ons = [o for i,o in enumerate(d_ons) if i==0 or o-d_ons[i-1]>=2]
S = np.abs(librosa.stft(yd,n_fft=2048,hop_length=256))
freqs = librosa.fft_frequencies(sr=SR,n_fft=2048)
ke = S[(freqs>=20)&(freqs<=200)].mean(axis=0)
se = S[(freqs>=300)&(freqs<=5000)].mean(axis=0)
he = S[(freqs>=6000)].mean(axis=0)

drum_seq=[]
for o in d_ons:
    lo,hi=max(0,o-1),min(len(ke),o+3)
    k=float(np.mean(ke[lo:hi])); s=float(np.mean(se[lo:hi])); h=float(np.mean(he[lo:hi]))
    mx=max(k,s,h)
    if mx<0.001: continue
    label = "dr_kick" if k==mx else ("dr_snare" if s==mx else "dr_hat")
    drum_seq.append((o*256/SR,label))

yb = librosa.util.normalize(librosa.load(STEMS/"Toter Schmetterling_bass.wav",sr=SR,mono=True)[0][:int(FOCUS*SR)])
f0,_,_=librosa.pyin(yb,fmin=30,fmax=500,sr=SR,frame_length=4096,hop_length=256)
pn=[hz2n(f) for f in f0]
b_ons=onsets(yb); b_ons=[o for i,o in enumerate(b_ons) if i==0 or o-b_ons[i-1]>=2]
bset={f.stem for f in BASS.glob("bass_*.wav")}
bass_seq=[]
for o in b_ons:
    lo,hi=max(0,o-1),min(len(pn),o+4)
    w=[n for n in pn[lo:hi] if n!="~"]
    if not w: continue
    note=Counter(w).most_common(1)[0][0]; key=f"bass_{note}"
    if key in bset: bass_seq.append((o*256/SR,key))

def pat(seq,spb=4):
    nb=int(FOCUS/BD)+1; ns=nb*spb; sd=BD/spb
    grid=[[] for _ in range(ns)]
    for t,k in seq:
        si=int(t/sd)
        if 0<=si<ns: grid[si].append(k)
    bars=[]
    for bs in range(0,ns,16):
        st=[]
        for si in range(bs,min(bs+16,ns)):
            ids=grid[si]
            st.append("~" if not ids else (ids[0] if len(ids)==1 else "["+" ".join(ids)+"]"))
        bars.append("["+" ".join(st)+"]")
    return " ".join(bars)

bp=pat(bass_seq); dp=pat(drum_seq)
cps=TEMPO/60.0

code = f"""samples('github:voglll/strudel-converter')
setcps({cps:.4f})

stack(
  // bass
  stack(`{bp}`).gain(0.9).lpf(500),
  // drums
  stack(`{dp}`).gain(0.85).room(0.05),
)
"""
OUT.write_text(code)
print(f"Bass: {len(bass_seq)} events, Drums: {len(drum_seq)} events")
print(code)
