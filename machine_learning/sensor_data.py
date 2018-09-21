import numpy as np


class SensorDatum:
    """Represents a single data point for a single sensor."""

    def __init__(self, acc_values, gyro_values):
        assert len(acc_values) == 3
        assert len(gyro_values) == 3

        self.acc = acc_values
        self.gyro = gyro_values


class SensorData:
    """Represents multiple data points for a single sensor."""

    def __init__(self, sensor_data):
        """Takes list of SensorDatum as input.

        Stores list of all acc and gyro values as lists of triplets
        """
        self.acc = np.array([sensor_datum.acc for sensor_datum in sensor_data])
        self.gyro = np.array([sensor_datum.gyro for sensor_datum in sensor_data])

    def get_all_axes(self):
        """Get an array consisting of all the data"""
        return np.concatenate([self.acc, self.gyro], axis=1)
