import _thread
import socket
import struct
import time
import sys
import numpy as np
from machine_learning import data_collection, sensor_data
from pynput.keyboard import Listener, Key, KeyCode

PORT = 8888
DATA_SIZE = 32


def register_kbd_listeners(on_move):
    def on_press(key):
        if key == Key.space:
            print('Move')
            on_move()

    Listener(on_press=on_press).start()


def recv_data(host):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, PORT))
    s.listen(1)

    conn, addr = s.accept()
    print ('Connected by', addr)

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

        contents = np.array([struct.unpack('>h', packet[i: i + 2])[0] for i in range(0, 32, 2)])
        yield contents / 100, inter_packet_time


if __name__ == '__main__':
    data_collection = data_collection.DataCollection('test.pb')

    try:
        register_kbd_listeners(on_move=data_collection.next_move)

        for packet, inter_packet_time in recv_data(sys.argv[1]):
            sensor1_datum = sensor_data.SensorDatum(packet[0:3], packet[3:6])
            sensor2_datum = sensor_data.SensorDatum(packet[6:9], packet[9:12])

            data_collection.process([sensor1_datum, sensor2_datum], inter_packet_time)

    except KeyboardInterrupt:
        data_collection.save()
        print('Done!')
