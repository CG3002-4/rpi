class SensorDatum:
    """Represents a single data point for a single sensor"""
    def __init__(self, acc_values, gyro_values):
        assert len(acc_values) == 3
        assert len(gyro_values) == 3

        self.acc = acc_values
        self.gyro = gyro_values
