import numpy as np
import namelist

import socket
# initialize socket
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


_prev_pixels = np.tile(253,(3,namelist.n_pixels))

pixels = np.tile(1,(3,namelist.n_pixels))

def send_pixel_up():
    global pixels, _prev_pixels

    pixels = np.clip(pixels,0,255).astype(int)


    MAX_PIXELS_PER_PACKET = 126

    idx = range(pixels.shape[1])
    idx = [i for i in idx if not np.array_equal(pixels[:,i],_prev_pixels[:,i])]
    #print(idx)
    n_packets = len(idx) // MAX_PIXELS_PER_PACKET + 1
    idx = np.array_split(idx,n_packets)
    for packet_indices in idx:
        m = []
        for i in packet_indices:
            m.append(i)
            m.append(pixels[0][i])
            m.append(pixels[1][i])
            m.append(pixels[2][i])
        m = bytes(m)
        _sock.sendto(m,(namelist.ip_addr,namelist.port_id))
    _prev_pixels = np.copy(pixels)


def update():
    send_pixel_up()


if __name__ == '__main__':
    import time 

    pixels *= 0
    pixels[0,0] = 255
    pixels[1,1] = 255
    pixels[2,2] = 255

    while True:
        pixels = np.roll(pixels,1,axis=1)
        update()
        time.sleep(.1)

