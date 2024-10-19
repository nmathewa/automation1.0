import time
import numpy as np
import pyaudio
import config

def start_stream(callback):
    p = pyaudio.PyAudio()
    frames_per_buffer = int(config.MIC_RATE / config.FPS)

    # Open the stream with optimized settings
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=config.MIC_RATE,
        input=True,
        frames_per_buffer=frames_per_buffer
    )

    overflows = 0
    prev_ovf_time = time.time()

    # Main loop for reading and processing audio data
    while True:
        try:
            # Non-blocking read to avoid delay
            y = np.frombuffer(stream.read(frames_per_buffer, exception_on_overflow=False), dtype=np.int16)
            y = y.astype(np.float32)

            # Avoid unnecessary reads (remove second stream.read)
            available = stream.get_read_available()
            if available > 0:
                stream.read(available, exception_on_overflow=False)

            # Pass the audio data to the callback function
            callback(y)

        except IOError:
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
                print(f'Audio buffer has overflowed {overflows} times')

    # Close the stream properly
    stream.stop_stream()
    stream.close()
    p.terminate()
