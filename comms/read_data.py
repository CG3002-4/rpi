import numpy as np
import serial
import struct

DATA_SIZE = 32


def read_data():
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    ser.flushInput()

    while True:
        packet = ser.read(DATA_SIZE + 1)

        checksum = 0
        for byte in packet[:DATA_SIZE]:
            checksum ^= byte

        if checksum != packet[DATA_SIZE]:
            print('Checksums didn\'t match')
            print(packet)
            print()

        # TODO: Should this be little endian??
        contents = np.array([struct.unpack('>h', packet[i: i + 2])[0] for i in range(0, 32, 2)])
        yield contents / 100
