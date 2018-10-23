

import serial
import struct
import numpy as np
import time

NUM_DATUM = 14
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
        packet = ser.readline()
        print(packet)

        data = np.array(packet.split(',')).astype(int)
        print(data)
        data_as_bytes = b''.join([struct.pack('>h', datum) for datum in data[:-1]])
        print(data_as_bytes)

        checksum = 0
        for byte in data_as_bytes:
            checksum ^= byte

        if checksum != data[NUM_DATUM]:
            print('Checksums didn\'t match')
            print()

        yield data / 100
