"""Code to be run on RPi while collecting data.

Usage:
    python3 rpi_collect.py host_ip port
"""
import sys
from data_collection import DataCollection
from recv_data import recv_data
import sensor_data


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
