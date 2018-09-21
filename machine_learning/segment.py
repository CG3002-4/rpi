SEGMENT_SIZE = 50
SEGMENT_OVERLAP = 0
SEGMENT_OFFSET = int(SEGMENT_SIZE * (1 - SEGMENT_OVERLAP))


class Segment:
    """Represents a segment of data, containing a fixed number of data points
    per sensor.
    """
    def __init__(self, sensors_data, label):
        assert len(sensors_data) == SEGMENT_SIZE

        self.sensors_data = sensors_data
        self.label = label

    def __repr__(self):
        return str(self.label)
