from Crypto.Cipher import AES
from Crypto import Random

import base64
import sys
import os
import socket


PORT = 8888
DATA_SIZE = 32
secret_key = '1234567887654321'


BS = 16
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)

#Add handshaking code

def read_and_analyse_data():
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    ser.flushInput()

    while True:
        packet = ser.read(DATA_SIZE + 1)

        checksum = 0
        for byte in packet[:DATA_SIZE]:
            checksum ^= byte

        if checksum == packet[DATA_SIZE]:
            print('Checksum match')
            contents = np.array([struct.unpack('>h', packet[i: i + 2])[0] for i in range(0, 32, 2)])
            yield contents / 100
        else:
            print('Checksums didn\'t match')
            print(packet)
            print()

def encryptText(msg, Key):
    iv = Random.new().read(AES.block_size)
    secret_key = bytes(str(Key), encoding = "utf8")
    cipher = AES.new(secret_key,AES.MODE_CBC,iv)
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
        5: "turnclap"
        11: "logout"
    }
    return switcher.get(action_num, "Invalid move")
        

def send_data(s, action_num, voltage, current, power, cumpower):
    action = switch_action(action_num)
    results = "#" + action + "|" + str(voltage) + "|" + str(current) + "|" + str(power) + "|" + str(cumpower)
    encodedResults = encryptText(results, secret_key)
    s.sendall(encodedResults)
        
    if action == "logout":
        return -1

def logout(s):
    print("Bye!")
    s.close()
        