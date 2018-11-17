import numpy as np
import os
import segment
import sensor_data as sd


NUM_SENSORS = 2


class DataCollection:
    """Convenience class implementing data collections.

    Expected usage:
        1. Create new instance
        2. On each new data point, call process()
        3. When a move finishes (use button interrupts on RPi?) call next_move()
        4. Save object to file
        5. Manually record labels for each of the next_move calls
        6. Load object from file
        7. Call segment() using labels
    """

    def __init__(self, experiment_dir):
        self.sensors_data = np.empty((0, 6 * NUM_SENSORS))
        self.experiment_dir = experiment_dir
        self.label = None

    def process(self, sensors_datum):
        """
        Takes in a list of sensor_data.SensorDatum representing one data
        point for each sensor.
        """
        assert len(sensors_datum) == 6 * NUM_SENSORS

        self.sensors_data = np.append(self.sensors_data, np.array([sensors_datum]), axis=0)

    def save(self):
        if not os.path.isdir(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        def dump_csv(data, filename, fmt):
            file_location = os.path.join(self.experiment_dir, filename)
            np.savetxt(file_location,
                       data, delimiter=',', fmt=fmt)

        for i in range(NUM_SENSORS):
            dump_csv(self.sensors_data[:, 6 * i: 6 * i + 3], 'sensor' + str(i) + '_acc.txt', '% f')
            dump_csv(self.sensors_data[:, 6 * i + 3: 6 * i + 6], 'sensor' + str(i) + '_gyro.txt', '% f')

        if self.label is not None:
            dump_csv([self.label], 'labels.txt', '%d')

    def load(self):
        def load_csv(filename, dtype):
            file_location = os.path.join(self.experiment_dir, filename)
            return np.loadtxt(file_location, delimiter=',', dtype=dtype)

        self.sensors_data = None

        for i in range(NUM_SENSORS):
            acc = load_csv('sensor' + str(i) + '_acc.txt', dtype=float)
            gyro = load_csv('sensor' + str(i) + '_gyro.txt', dtype=float)
            if self.sensors_data is not None:
                self.sensors_data = np.hstack([self.sensors_data, acc, gyro])
            else:
                self.sensors_data = np.hstack([acc, gyro])

        self.label = load_csv('labels.txt', dtype=int)

    def segment(self):
        """Given labels for each of the moves, segment the data.

        Note that this function simply segments the data according to the
        segment size and overlap values in segment.py.

        It then labels each segment by checking which label accounts for
        the majority of the segment.
        """
        assert self.label is not None

        segment_start = 0
        segments = []
        while segment_start + segment.SEGMENT_SIZE <= len(self.sensors_data):
            segment_data = self.sensors_data[segment_start: segment_start + segment.SEGMENT_SIZE]
            segments.append(segment.Segment(segment_data, self.label))

            segment_start += segment.SEGMENT_OFFSET

        return segments


if __name__ == '__main__':
    # Test the DataCollection class
    import random

    def random_array(length, low, high):
        return [random.randrange(low, high) for i in range(length)]

    NUM_MOVES = 10
    NUM_LABELS = 12
    EXP_LOCATION = os.path.join('data_test', 'test_exp')
    NUM_DATA_POINTS = segment.SEGMENT_SIZE * 30

    # Generate random sensor data
    sensors_data = [random_array(NUM_SENSORS * 6, 0, 10) for i in range(NUM_DATA_POINTS)]

    # Process each data point
    data_collection = DataCollection(EXP_LOCATION)
    for datum in sensors_data:
        data_collection.process(datum)

    data_collection.label = 1

    # Save and reload
    data_collection.save()
    data_collection = DataCollection(EXP_LOCATION)
    data_collection.load()

    # Segment
    segments = data_collection.segment()
    print(data_collection.label)
    print(segments)
