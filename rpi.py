import socket
import serial
import sys

PORT = 8888
DATA_SIZE = 32


def read_and_send_data(host):
    ser = serial.Serial("/dev/ttyAMA0", 115200)
    ser.flushInput()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, PORT))

    while True:
        packet = ser.read(DATA_SIZE + 1)

        checksum = 0
        for byte in packet[:DATA_SIZE]:
            checksum ^= byte

        if checksum != packet[DATA_SIZE]:
            print('Checksums didn\'t match')
            print(packet)
            print()

        s.sendall(packet)


if __name__ == '__main__':
    read_and_send_data(sys.argv[1])
