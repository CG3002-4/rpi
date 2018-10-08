import numpy as np


NUM_AXES = 3


class ProcessedSensorData:
    """Represents multiple processed data points for a single sensor."""

    def __init__(self, body_values, grav_values, gyro_values):
        """Expects input data to be numpy arrays"""
        assert body_values.shape[1] == NUM_AXES
        assert grav_values.shape[1] == NUM_AXES
        assert gyro_values.shape[1] == NUM_AXES

        self.body = body_values
        self.grav = grav_values
        self.gyro = gyro_values

    def get_all_axes(self):
        """Get an array consisting of all the data"""
        return np.concatenate([self.body, self.grav, self.gyro], axis=1)
