import numpy as np


NUM_AXES = 3


class ProcessedData:
    """Represents multiple processed data points for a single sensor."""

    def __init__(self, body_values, grav_values, gyro_values):
        self.body = body_values
        self.grav = grav_values
        self.gyro = gyro_values
