import serial
import struct

ser = serial.Serial("/dev/ttyAMA0", 115200)
ser.flushInput()

while True:
    packet = ser.read(33)
    print(packet)

    checksum = 0
    for byte in packet[:32]:
        checksum ^= byte

    if checksum != packet[32]:
        print('Oh no')
    # print(checksum)
    # print(packet[32])

    for i in range(0, 32, 2):
        print(struct.unpack('>h', packet[i: i + 2])[0])
    print()
