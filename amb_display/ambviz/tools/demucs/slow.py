"""The director does not consume per-frame envelopes. It consumes slow ones.

`mood.response_seconds` and the score smoothing run over seconds, and switches
are 8-90 s apart. So the question is not whether a stale mask tracks a kick,
but whether it tracks "how much drum is in this passage" over a scene.
"""
import numpy as np, torch, torchaudio, wave
dev="cuda"; bundle=torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
model=bundle.get_model().to(dev).eval(); SR=bundle.sample_rate; SRC=model.sources
w=wave.open("capture.wav"); sr=w.getframerate()
raw=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16)
wav=torch.from_numpy(raw.reshape(-1,w.getnchannels()).T.astype(np.float32)/2**15)
if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)
wav=wav[:,:int(30*SR)]
with torch.no_grad(): stems=model(wav.unsqueeze(0).to(dev))[0].cpu()
N_FFT,HOP=2048,735; win=torch.hann_window(N_FFT)
f=lambda x: torch.stft(x.mean(0),N_FFT,HOP,window=win,return_complex=True).abs().numpy()
S=np.stack([f(s) for s in stems]); MIX=f(wav); F=MIX.shape[1]; truth=S.sum(axis=1)
FR=SR/HOP

def smooth(x, sec):
    n=max(1,int(sec*FR)); k=np.ones(n)/n
    return np.stack([np.convolve(r,k,mode="same") for r in x])

# stale mask: 1 s window, 1 s lag -- comfortably achievable, 19 ms of GPU
wf=lf=int(1.0*FR)
est=np.zeros((4,F))
for i in range(wf+lf,F):
    mm=S[:,:,max(0,i-lf-wf):i-lf+1].mean(axis=2)
    mm=mm/np.maximum(mm.sum(0,keepdims=True),1e-9)
    est[:,i]=(mm*MIX[:,i]).sum(1)
sl=slice(wf+lf,F)

print("stale mask (1 s window, 1 s lag), correlation after smoothing both sides\n")
print(f"{'smoothing':>10}   mean   per stem")
for sec in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
    a = est[:,sl] if sec==0 else smooth(est,sec)[:,sl]
    b = truth[:,sl] if sec==0 else smooth(truth,sec)[:,sl]
    cs=[np.corrcoef(a[k],b[k])[0,1] for k in range(4)]
    print(f"{sec:8.1f}s   {np.mean(cs):.3f}   " +
          " ".join(f"{n[:5]}={c:.2f}" for n,c in zip(SRC,cs)))

# what the director would actually read: stem *balance*, not absolute level
print("\nstem balance (share of total), 2 s smoothing:")
ea=smooth(est,2.0)[:,sl]; ta=smooth(truth,2.0)[:,sl]
ea=ea/np.maximum(ea.sum(0,keepdims=True),1e-9); ta=ta/np.maximum(ta.sum(0,keepdims=True),1e-9)
for k,n in enumerate(SRC):
    print(f"  {n:<7} corr={np.corrcoef(ea[k],ta[k])[0,1]:.3f}   "
          f"mean abs error={np.abs(ea[k]-ta[k]).mean():.3f}")
