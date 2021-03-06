SEGMENT_SIZE = 100
SEGMENT_OVERLAP = 0.9
SEGMENT_OFFSET = int(SEGMENT_SIZE * (1 - SEGMENT_OVERLAP))


class Segment:
    """Represents a segment of data, containing a fixed number of data points
    per sensor.

    The segment is represented as a list of either sensor_data.SensorData or processed_sensor_data.ProcessedData,
    i.e. for each sensor, the relevant data is put together.
    """

    def __init__(self, sensors_data, label):
        assert sensors_data.shape == (SEGMENT_SIZE, 12)

        self.sensors_data = sensors_data
        self.label = label

    def __repr__(self):
        return str(self.label)

    def __str__(self):
        return str(self.sensors_data)
