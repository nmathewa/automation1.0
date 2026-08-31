"""Separate two error sources that the last run conflated.

  staleness   -- the mask is old
  representation -- a per-bin scalar mask cannot express what Demucs does

The control is a mask computed from the CURRENT frame. If that is still bad,
staleness is not the problem and no amount of GPU speed will help.
"""
import numpy as np, torch, torchaudio, wave

dev="cuda"
bundle=torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
model=bundle.get_model().to(dev).eval(); SR=bundle.sample_rate; SOURCES=model.sources
w=wave.open("capture.wav"); sr=w.getframerate()
raw=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16)
wav=torch.from_numpy(raw.reshape(-1,w.getnchannels()).T.astype(np.float32)/2**15)
if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)
wav=wav[:,:int(30*SR)]
with torch.no_grad(): stems=model(wav.unsqueeze(0).to(dev))[0].cpu()

N_FFT,HOP=2048,735; win=torch.hann_window(N_FFT)
f=lambda x: torch.stft(x.mean(0),N_FFT,HOP,window=win,return_complex=True).abs().numpy()
S=np.stack([f(s) for s in stems]); MIX=f(wav); F=MIX.shape[1]
truth=S.sum(axis=1)

def report(label, est, sl):
    e=np.abs(est[:,sl]-truth[:,sl]).sum()/max(truth[:,sl].sum(),1e-9)
    cs=[np.corrcoef(est[k,sl],truth[k,sl])[0,1] for k in range(4)]
    print(f"{label:<34} {100*e:5.1f}%   " + "  ".join(f"{n[:5]}={c:.3f}" for n,c in zip(SOURCES,cs)))

print(f"{'':<34} {'err':>6}   correlation per stem")

# ORACLE: mask from this very frame -- the ceiling any masking scheme can reach
m=S/np.maximum(S.sum(0,keepdims=True),1e-9)
report("oracle mask, zero lag", (m*MIX[None]).sum(1), slice(0,F))

# stale masks, for contrast
for lag_s in (0.25, 0.5, 1.0, 2.0):
    lf=int(lag_s*SR/HOP); wf=int(1.0*SR/HOP)
    est=np.zeros((4,F))
    for i in range(wf+lf,F):
        a=max(0,i-lf-wf); mm=S[:,:,a:i-lf+1].mean(axis=2)
        mm=mm/np.maximum(mm.sum(0,keepdims=True),1e-9)
        est[:,i]=(mm*MIX[:,i]).sum(1)
    report(f"1.0s window, {lag_s:.2f}s lag", est, slice(wf+lf,F))

# how fast does the routing itself actually change?
mt=m/np.maximum(m.sum(0,keepdims=True),1e-9)
for gap_s in (0.25,1.0,2.0):
    g=int(gap_s*SR/HOP)
    d=np.abs(mt[:,:,g:]-mt[:,:,:-g]).mean()
    print(f"mean |mask(t) - mask(t-{gap_s:.2f}s)| = {d:.3f}  (0 = routing is static)")
