import os
# get path for current file


DEVICE = 'esp8266'

if DEVICE == 'esp8266':
    UDP_IP = '192.168.0.150'
    # IP should match IP in ws2812_controller.ino
    UDP_PORT = 7777
    SOFTWARE_GAMMA_CORRECTION = False

else :
    raise Exception("Device not supported")

N_PIXELS = 60
MIC_RATE = 44100
FPS = 60
MIN_FREQUENCY = 200
MAX_FREQUENCY = 12000
N_FFT_BINS = 24
N_ROLLING_HISTORY = 2
MIN_VOLUME_THRESHOLD = 1e-7
GAMMA_TABLE_PATH = os.path.dirname(__file__) +'/esp_led/gamma_table.npy'
DISPLAY_FPS = True
