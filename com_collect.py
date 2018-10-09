"""Code to be run on computer while collecting data.

Usage:
    python3 com_collect.py host_ip port experiment_dir
"""
import socket
import struct
import time
import sys
import numpy as np
import data_collection
import sensor_data
from pynput.keyboard import Listener, Key

DATA_SIZE = 32


def register_kbd_listeners(on_move):
    def on_press(key):
        if key == Key.space:
            print('Move')
            on_move()

    Listener(on_press=on_press).start()


def recv_data(host_ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host_ip, port))
    s.listen(1)

    conn, addr = s.accept()
    print('Connected by', addr)

    prev_time = time.time()

    while (True):
        packet = conn.recv(DATA_SIZE + 1)
        curr_time = time.time()
        inter_packet_time = curr_time - prev_time

        prev_time = curr_time

        checksum = 0
        for byte in packet[:DATA_SIZE]:
            checksum ^= byte

        if checksum != packet[DATA_SIZE]:
            print('Checksums didn\'t match')
            print(packet)
            print()

        contents = np.array([struct.unpack('>h', packet[i: i + 2])[0]
                             for i in range(0, 32, 2)])
        yield contents / 100, inter_packet_time


if __name__ == '__main__':
    data_collection = data_collection.DataCollection(experiment_dir=sys.argv[3])
    data_collection.next_move()

    try:
        register_kbd_listeners(on_move=data_collection.next_move)

        for unpacked_data, inter_packet_time in recv_data(host_ip=sys.argv[1], port=int(sys.argv[2])):
            sensor1_datum = sensor_data.SensorDatum(
                unpacked_data[0:3], unpacked_data[3:6])
            sensor2_datum = sensor_data.SensorDatum(
                unpacked_data[6:9], unpacked_data[9:12])

            data_collection.process(
                [sensor1_datum, sensor2_datum], inter_packet_time)

    except KeyboardInterrupt:
        data_collection.save()
        print('Done!')
