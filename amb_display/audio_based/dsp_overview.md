# LED Audio Sync with Adaptive DSP Techniques

## 1. Goal: Adaptive LED Control for Live Audio

We want to build a real-time LED system that dynamically adjusts based on the audio input. The goal is to:

- Detect dominant frequencies in the audio.
- Adapt filtering to avoid sudden shifts caused by vocals.
- Focus on background harmonic elements (e.g., bass, chords) for smoother visualizations.

## 2. Key Challenges Identified

- **Handling Vocals and Sudden Frequency Shifts:**
  - Fast frequency changes (e.g., in vocals) can cause flickering in LED animations.
- **Computational Constraints:**
  - Moving averages or complex smoothing techniques may not work efficiently for real-time audio.

## 3. Initial Approach: Dynamic Bandpass Filter

We started by implementing a dynamic bandpass filter that:

- Detects the dominant frequency using FFT.
- Shifts the bandpass filter range dynamically to track the interesting frequency band.
- **Challenges:** Sudden frequency changes from vocals caused instability.

### Code Overview: Dynamic Bandpass Filter with FFT

- Uses real-time FFT analysis to determine the dominant frequency.
- Applies a bandpass filter centered on that frequency.
- Adjusts LED patterns based on the energy of the filtered signal.

## 4. Shift to Lightweight Vocal Suppression (Spectral Filtering Approach)

**Objective:** Prioritize smoother, harmonic content while reducing vocal influence.

We identified Method 2: Harmonic Content Emphasis as a more effective strategy:

- **Spectral Rolloff:** Measures the frequency below which most of the spectral energy is concentrated.
- Vocals tend to have energy spread across higher frequencies.
- Harmonic sounds (like bass and chords) exhibit smoother, more stable patterns with energy concentrated in the lower frequencies.

## 5. Algorithm: Spectral Rolloff-Based Adaptive Filtering

### Algorithm Overview:

1. **Compute Spectral Rolloff:**
   - Use FFT to compute the cumulative energy distribution.
   - Find the rolloff frequency where 85% of the energy is below that point.

2. **Adaptive Filtering:**
   - If rolloff > 1000 Hz, apply a lowpass filter to suppress vocals.
   - If rolloff < 1000 Hz, pass the signal through as-is.

3. **LED Mapping:**
   - Use the energy of the filtered signal to control LED brightness and animations.

## 6. Code: Spectral Rolloff Filtering for Live Audio

```python
import numpy as np
import pyaudio
from scipy.fftpack import fft
from scipy.signal import get_window, butter, sosfilt

def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sos

def apply_lowpass_filter(data, cutoff, fs, order=4):
    sos = butter_lowpass(cutoff, fs, order)
    return sosfilt(sos, data)

def audio_stream(chunk_size=1024, sample_rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)
    return stream

def compute_spectral_rolloff(data, sample_rate, roll_percent=0.85):
    N = len(data)
    windowed_data = data * get_window('hann', N)
    spectrum = np.abs(fft(windowed_data)[:N // 2])
    cumulative_energy = np.cumsum(spectrum)
    total_energy = cumulative_energy[-1]
    rolloff_idx = np.where(cumulative_energy >= roll_percent * total_energy)[0][0]
    freqs = np.fft.fftfreq(N, 1 / sample_rate)[:N // 2]
    return freqs[rolloff_idx]

def main():
    CHUNK = 1024
    RATE = 44100
    stream = audio_stream(CHUNK, RATE)
    print("Streaming audio... Press Ctrl+C to stop.")

    try:
        while True:
            data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
            rolloff_freq = compute_spectral_rolloff(data, RATE)
            print(f"Spectral Rolloff Frequency: {rolloff_freq:.2f} Hz")

            if rolloff_freq > 1000:
                print("Vocals detected – Applying lowpass filter")
                filtered_data = apply_lowpass_filter(data, 500, RATE)
            else:
                print("Smooth background detected")
                filtered_data = data

            energy = np.sum(filtered_data ** 2)
            print(f"Filtered Signal Energy: {energy}")

    except KeyboardInterrupt:
        print("Stopping stream...")
        stream.stop_stream()
        stream.close()

if __name__ == "__main__":
    main()

```

# 7. Why This Approach Works for Real-Time Audio
- Lightweight and Fast: FFT and rolloff calculation are computationally efficient.
- Resistant to Sudden Changes: By focusing on spectral rolloff, it ignores transient    spikes from vocals.
- Background Emphasis: Lowpass filtering ensures that harmonic sounds are prioritized - for smoother visualizations.
# 8. Further Enhancements
- Dynamic Threshold Adjustment:
Adjust the rolloff threshold (1000 Hz) based on the audio profile.
- Multi-Band LED Control:
Split the signal into bass, mid, and high frequencies to drive multiple LED strips.
- Beat Detection Integration:
Combine this with beat detection to make LEDs pulse to the rhythm.