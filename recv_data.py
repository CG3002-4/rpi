

import serial
import struct
import numpy as np
import time

DATA_SIZE = 28
HANDSHAKING = False

def recv_data():
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    ser.flushInput()
    if HANDSHAKING:
        time.sleep(5)
        ser.write("4".encode('utf8'))
        ack = ser.read(1)
    
        if (ack[0] == 6):
            ser.write("3".encode('utf8'))
            print("Handshaking success")
        else:
            assert False, "Handshaking failed"
    
    print('Looking for data')
    while True:
        packet = ser.read(DATA_SIZE + 1)

        checksum = 0
        for byte in packet[:DATA_SIZE]:
            checksum ^= byte

        if checksum != packet[DATA_SIZE]:
            print('Checksums didn\'t match')
            print(packet)
            print()

        contents = np.array([struct.unpack('>h', packet[i: i + 2])[0]
                             for i in range(0, DATA_SIZE, 2)])
        yield contents / 100
    

    
