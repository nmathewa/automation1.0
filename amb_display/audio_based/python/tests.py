#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:23:14 2024

@author: nalex2023
"""

import numpy as np
import pyaudio
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore  
import struct
#%%

gamma_table = '/Users/nalex2023/main/automation1.0/amb_display/audio_based/python/gamma_table.npy'

CHUNK = 1024            # Number of audio samples per frame
RATE = 48000            # Sample rate in Hz
GAMMA_TABLE = np.load(gamma_table)

#%%

import numpy as np
import pyaudio
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import struct

CHUNK = 1024
RATE = 44100
GAMMA_TABLE = np.load(gamma_table)  # Load gamma correction table

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Live Audio Visualizer")
win.show()

waveform_plot = win.addPlot(title="Waveform")
waveform_curve = waveform_plot.plot(pen='c')

win.nextRow()
spectrum_plot = win.addPlot(title="Frequency Spectrum")
spectrum_curve = spectrum_plot.plot(pen='m')

alpha = 0.2  # Smoothing factor
prev_fft_magnitude = np.zeros(CHUNK // 2 + 1)

def gamma_correct(value, gamma_table):
    index = int(min(max(value, 0), len(gamma_table) - 1))
    return gamma_table[index]

def smooth_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def smooth_fft(current_fft):
    global prev_fft_magnitude
    smoothed = alpha * current_fft + (1 - alpha) * prev_fft_magnitude
    prev_fft_magnitude = smoothed
    return smoothed

def to_decibel(magnitude):
    return 20 * np.log10(np.maximum(magnitude, 1e-10))

def update():
    data = stream.read(CHUNK, exception_on_overflow=False)
    data_int = np.array(struct.unpack(f'{CHUNK}h', data), dtype=np.int16)

    waveform = np.array([gamma_correct(val, GAMMA_TABLE) for val in data_int])
    waveform_smoothed = smooth_data(waveform)
    waveform_curve.setData(waveform_smoothed, autoDownsample=True)

    fft_data = np.fft.rfft(waveform_smoothed)
    fft_magnitude = np.abs(fft_data)
    fft_magnitude_smoothed = smooth_fft(fft_magnitude)
    fft_magnitude_db = to_decibel(fft_magnitude_smoothed)

    spectrum_curve.setData(fft_magnitude_db, autoDownsample=True)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)  # Faster refresh rate for smoother animations

print("Starting live audio visualizer...")
app.exec_()

stream.stop_stream()
stream.close()
p.terminate()
