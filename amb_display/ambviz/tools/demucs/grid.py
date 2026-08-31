"""Window length and lag are different things. Which one actually kills it?"""
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
FR=SR/HOP   # 60 frames/s

print(f"lag is in frames of {1000/FR:.1f} ms\n")
print(f"{'window':>8} {'lag':>8}  {'err':>6}   mean corr   per stem")
for win_f, lag_f in [(1,0),(1,1),(1,3),(1,6),(1,15),(1,30),(1,60),
                     (3,3),(6,6),(15,15),(60,15)]:
    est=np.zeros((4,F)); start=win_f+lag_f
    for i in range(start,F):
        a=max(0,i-lag_f-win_f+1); b=i-lag_f+1
        mm=S[:,:,a:b].mean(axis=2)
        mm=mm/np.maximum(mm.sum(0,keepdims=True),1e-9)
        est[:,i]=(mm*MIX[:,i]).sum(1)
    sl=slice(start,F)
    e=np.abs(est[:,sl]-truth[:,sl]).sum()/max(truth[:,sl].sum(),1e-9)
    cs=[np.corrcoef(est[k,sl],truth[k,sl])[0,1] for k in range(4)]
    print(f"{win_f*1000/FR:6.0f}ms {lag_f*1000/FR:6.0f}ms  {100*e:5.1f}%   {np.mean(cs):.3f}    " +
          " ".join(f"{n[:5]}={c:.2f}" for n,c in zip(SRC,cs)))
