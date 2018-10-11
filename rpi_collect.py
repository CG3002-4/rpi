"""Code to be run on RPi while collecting data.

Usage:
    python3 rpi_collect.py host_ip port
"""
import serial
import sys
import struct
import numpy as np
from data_collection import DataCollection
import sensor_data

DATA_SIZE = 32


def recv_data():
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    ser.flushInput()

    while True:
        packet = ser.read(DATA_SIZE + 1)

        checksum = 0
        for byte in packet[:DATA_SIZE]:
            checksum ^= byte

        if checksum != packet[DATA_SIZE]:
            print('Checksums didn\'t match')
            print()

        contents = np.array([struct.unpack('>h', packet[i: i + 2])[0]
                             for i in range(0, 32, 2)])
        yield contents / 100


if __name__ == '__main__':
    data_collection = DataCollection(experiment_dir=sys.argv[1])
    data_collection.next_move()

    try:
        for unpacked_data in recv_data():
            sensor1_datum = sensor_data.SensorDatum(
                unpacked_data[0:3], unpacked_data[3:6])
            sensor2_datum = sensor_data.SensorDatum(
                unpacked_data[6:9], unpacked_data[9:12])

            data_collection.process([sensor1_datum, sensor2_datum])
    except KeyboardInterrupt:
        # Use second argument as label for entire data
        data_collection.labels = [int(sys.argv[2])]
        data_collection.save()
