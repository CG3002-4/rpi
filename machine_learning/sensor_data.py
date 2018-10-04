import numpy as np


NUM_AXES = 3


class SensorDatum:
    """Represents a single data point for a single sensor."""

    def __init__(self, acc_values, gyro_values):
        assert len(acc_values) == NUM_AXES
        assert len(gyro_values) == NUM_AXES

        self.acc = acc_values
        self.gyro = gyro_values


class SensorData:
    """Represents multiple data points for a single sensor."""

    def __init__(self):
        """Stores list of all acc and gyro values as lists of triplets"""
        self.acc = np.empty((0, NUM_AXES))
        self.gyro = np.empty((0, NUM_AXES))

    def add_datum(self, sensor_datum):
        self.acc = np.vstack([self.acc, sensor_datum.acc])
        self.gyro = np.vstack([self.gyro, sensor_datum.gyro])

    def get_slice(self, start, stop):
        sliced = SensorData()
        sliced.acc = self.acc[start:stop]
        sliced.gyro = self.gyro[start:stop]
        return sliced

    def set_data(self, acc, gyro):
        assert acc.shape == gyro.shape
        assert acc.shape[1] == NUM_AXES

        self.acc = acc
        self.gyro = gyro

    def get_all_axes(self):
        """Get an array consisting of all the data"""
        return np.concatenate([self.acc, self.gyro], axis=1)
