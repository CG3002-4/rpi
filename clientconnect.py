from Crypto.Cipher import AES
from Crypto import Random

import numpy as np
import base64
import sys
import os
import socket
import time
import serial
import struct

PORT = 8888
NUM_DATUM = 14
HANDSHAKING = False
SECRET_KEY = '1234567887654321'


BS = 16
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)


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

        try:
            packet = packet.decode('utf8')
            data = np.array(packet.strip('\r\n').split(',')).astype(int)
            data_as_bytes = b''.join([struct.pack('>h', datum) for datum in data[:-1]])

            checksum = 0
            for byte in data_as_bytes:
                checksum ^= byte

            if checksum != data[NUM_DATUM]:
                print('Checksums didn\'t match')
                print()

            yield data[:-1] / 100
        except:
            print("Failed to decode packet:")
            print(packet)
            print()


def encryptText(msg, Key):
    iv = Random.new().read(AES.block_size)
    secret_key = bytes(str(Key), encoding="utf8")
    cipher = AES.new(secret_key, AES.MODE_CBC, iv)
    encryptedText = cipher.encrypt(pad(msg))
    encodedText = base64.b64encode(iv + encryptedText)

    return encodedText


def create_socket(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print("Socket connected")

    return s


def switch_action(action_num):
    switcher = {
        1: "wipers",
        2: "number7",
        3: "chicken",
        4: "sidestep",
        5: "turnclap",
        11: "logout"
    }

    return switcher.get(action_num, None)


def send_data(s, action_num, voltage, current, power, cumpower):
    action = switch_action(action_num)

    if action is not None:
        results = "#" + action + "|" + str(voltage) + "|" + str(current) + "|" + str(power) + "|" + str(cumpower)
        encodedResults = encryptText(results, SECRET_KEY)
        s.sendall(encodedResults)

        if action == "logout":
            return -1


def logout(s):
    print("Bye!")
    s.close()
