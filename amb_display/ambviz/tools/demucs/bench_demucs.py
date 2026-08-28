"""Does htdemucs keep up with real time on this 4050, and at what chunk size?

The question is not "is it fast" but two separate ones:
  throughput -- can it process 1 s of audio in under 1 s? (decides feasibility)
  latency    -- how long before a sample influences the mask? (decides design)
"""
import time, sys
import torch, torchaudio

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  device={dev}")
if dev == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name}  {p.total_memory/2**20:.0f} MiB  sm_{p.major}{p.minor}")
print()

bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().to(dev).eval()
sr = bundle.sample_rate
print(f"model: Hybrid Demucs, sources={model.sources}, sample_rate={sr}")
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
print()

torch.cuda.empty_cache() if dev == "cuda" else None
base = torch.cuda.memory_allocated() if dev == "cuda" else 0

print(f"{'chunk':>7} {'infer':>9} {'xRT':>7} {'VRAM peak':>10}  verdict")
for seconds in (0.5, 1.0, 2.0, 4.0, 8.0):
    x = torch.randn(1, 2, int(seconds * sr), device=dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(2):                       # warm up / autotune
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        N = 5
        for _ in range(N):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / N
    peak = (torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else 0
    xrt = seconds / dt
    ok = "real-time OK" if xrt > 1.0 else "TOO SLOW"
    print(f"{seconds:6.1f}s {dt*1000:8.1f}ms {xrt:6.1f}x {peak:9.0f}M  {ok}")
    del x
    if dev == "cuda":
        torch.cuda.empty_cache()
