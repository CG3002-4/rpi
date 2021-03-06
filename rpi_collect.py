"""Code to be run on RPi while collecting data.

Usage:
    python3 rpi_collect.py host_ip port
"""
import sys
from data_collection import DataCollection
import sensor_data
from clientconnect import recv_data
# from dummy_data import recv_data


if __name__ == '__main__':
    data_collection = DataCollection(experiment_dir=sys.argv[1])

    try:
        for unpacked_data in recv_data():
            data_collection.process(unpacked_data[:12])
    except KeyboardInterrupt:
        # Use second argument as label for data
        data_collection.label = [int(sys.argv[2])]
        data_collection.save()
