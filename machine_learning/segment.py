import numpy as np
import sensor_data

SEGMENT_SIZE = 50
SEGMENT_OVERLAP = 0
SEGMENT_OFFSET = int(SEGMENT_SIZE * (1 - SEGMENT_OVERLAP))


class Segment:
    """Represents a segment of data, containing a fixed number of data points
    per sensor.

    The segment is represented as a list of sensor_data.SensorData, i.e. for each
    sensor, the relevant data is put together.
    """
    def __init__(self, sensors_data, label):
        """sensors_data must be a numpy array"""
        assert len(sensors_data) == SEGMENT_SIZE

        self.sensors_data = np.array([sensor_data.SensorData(sd) for sd in sensors_data.transpose()])
        self.label = label

    def __repr__(self):
        return str(self.label)

    def __str__(self):
        return str(self.sensors_data)
