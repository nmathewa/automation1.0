"""Does a stale Demucs mask recover live per-stem envelopes on real music?

The claim under test: run the model late, apply its routing to the frame that
just arrived, and the per-stem energy still tracks the truth with no added
latency. Ground truth here is Demucs run on the *same* frame -- so this
measures the cost of staleness alone, not the model's separation quality.
"""
import time
import numpy as np, torch, torchaudio

dev = "cuda"
bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().to(dev).eval()
SR = bundle.sample_rate
SOURCES = model.sources

# torchaudio 2.11 delegates loading to torchcodec; `wave` needs no new deps.
import wave as _wave
_w = _wave.open("capture.wav")
sr = _w.getframerate()
raw = np.frombuffer(_w.readframes(_w.getnframes()), dtype=np.int16)
wav = torch.from_numpy(raw.reshape(-1, _w.getnchannels()).T.astype(np.float32) / 2**15)
if sr != SR:
    wav = torchaudio.functional.resample(wav, sr, SR)
wav = wav[:, :int(30 * SR)]
print(f"audio: {wav.shape[1]/SR:.1f}s  rms={wav.pow(2).mean().sqrt():.4f}")

t0 = time.perf_counter()
with torch.no_grad():
    stems = model(wav.unsqueeze(0).to(dev))[0].cpu()      # (4, 2, N)
print(f"separated in {time.perf_counter()-t0:.2f}s -> {SOURCES}\n")

N_FFT, HOP = 2048, 735                                     # ambviz's own analysis
win = torch.hann_window(N_FFT)
def stft_mag(x):
    return torch.stft(x.mean(0), N_FFT, HOP, window=win,
                      return_complex=True).abs().numpy()    # (bins, frames)

S = np.stack([stft_mag(s) for s in stems])                 # (4, bins, frames)
MIX = stft_mag(wav)
frames = MIX.shape[1]
truth = S.sum(axis=1)                                      # (4, frames) per-stem energy

def masks_at(i, win_frames):
    """Mask from a window of length win_frames ending at frame i."""
    a = max(0, i - win_frames)
    w = S[:, :, a:i+1].mean(axis=2)                        # (4, bins)
    return w / np.maximum(w.sum(0, keepdims=True), 1e-9)

print(f"{'window':>8} {'mask lag':>9} {'err':>7}   correlation per stem")
for win_s, lag_s in ((1.0, 0.5), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (4.0, 2.0)):
    wf, lf = int(win_s*SR/HOP), int(lag_s*SR/HOP)
    est = np.zeros((4, frames))
    for i in range(wf + lf, frames):
        m = masks_at(i - lf, wf)                           # routing from the past
        est[:, i] = (m * MIX[:, i]).sum(1)                 # energy from now
    sl = slice(wf + lf, frames)
    e = np.abs(est[:, sl] - truth[:, sl]).sum() / max(truth[:, sl].sum(), 1e-9)
    cs = [np.corrcoef(est[k, sl], truth[k, sl])[0, 1] for k in range(4)]
    print(f"{win_s:6.1f}s {lag_s:8.1f}s {100*e:6.1f}%   " +
          "  ".join(f"{n[:5]}={c:.3f}" for n, c in zip(SOURCES, cs)))
