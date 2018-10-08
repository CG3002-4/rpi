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
        self.sensors_data = np.array([sd.SensorData()
                                      for i in range(NUM_SENSORS)])
        self.inter_packet_times = []
        self.move_start_indices = []
        self.experiment_dir = experiment_dir
        self.num_data_points = 0
        self.labels = None

    def process(self, sensors_datum, inter_packet_time):
        """
        Takes in a list of sensor_data.SensorDatum representing one data
        point for each sensor, and inter packet time to compare latency.
        """
        assert len(sensors_datum) == NUM_SENSORS

        for i in range(NUM_SENSORS):
            self.sensors_data[i].add_datum(sensors_datum[i])

        self.inter_packet_times.append(inter_packet_time)
        self.num_data_points += 1

    def next_move(self):
        self.move_start_indices.append(self.num_data_points)

    def save(self):
        if not os.path.isdir(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        def dump_csv(data, filename, fmt):
            file_location = os.path.join(self.experiment_dir, filename)
            np.savetxt(file_location,
                       data, delimiter=',', fmt=fmt)

        for i, sensor_data in enumerate(self.sensors_data):
            dump_csv(sensor_data.acc, 'sensor' + str(i) + '_acc.txt', '% f')
            dump_csv(sensor_data.gyro, 'sensor' + str(i) + '_gyro.txt', '% f')

        dump_csv(self.inter_packet_times, 'inter_packet_times.txt', '%f')
        dump_csv(self.move_start_indices, 'move_start_indices.txt', '%d')

        if self.labels is not None:
            dump_csv(self.labels, 'labels.txt', '%d')

    def load(self):
        def load_csv(filename, dtype):
            file_location = os.path.join(self.experiment_dir, filename)
            return np.loadtxt(file_location, delimiter=',', dtype=dtype)

        for i in range(NUM_SENSORS):
            acc = load_csv('sensor' + str(i) + '_acc.txt', dtype=float)
            gyro = load_csv('sensor' + str(i) + '_gyro.txt', dtype=float)
            self.sensors_data[i].set_data(acc, gyro)

        self.inter_packet_times = list(
            load_csv('inter_packet_times.txt', dtype=float))
        self.move_start_indices = list(
            load_csv('move_start_indices.txt', dtype=int))
        self.num_data_points = len(self.sensors_data[0].acc)

        try:
            self.labels = np.array(load_csv('labels.txt', dtype=int))
        except:
            self.labels = None

    def segment(self):
        """Given labels for each of the moves, segment the data.

        Note that this function simply segments the data according to the
        segment size and overlap values in segment.py.

        It then labels each segment by checking which label accounts for
        the majority of the segment.
        """
        assert self.labels is not None
        assert len(self.labels) == len(self.move_start_indices)

        # To make segmenting a little easier
        self.move_start_indices.append(self.num_data_points)

        segment_start = 0
        curr_move_idx = 0
        segments = []
        while segment_start + segment.SEGMENT_SIZE <= self.num_data_points:
            # Compute what part of this segment is labelled by the current
            # index.
            portion_in_curr_move = self.move_start_indices[curr_move_idx +
                                                           1] - segment_start
            if portion_in_curr_move < segment.SEGMENT_SIZE // 2:
                # label with next move
                curr_move_idx += 1

            segment_data = np.array([sensor_data.get_slice(
                segment_start, segment_start + segment.SEGMENT_SIZE) for sensor_data in self.sensors_data])
            segments.append(segment.Segment(
                segment_data, self.labels[curr_move_idx]))

            segment_start += segment.SEGMENT_OFFSET

        # Undo mutation made to this array at the start
        self.move_start_indices.pop()

        return segments


if __name__ == '__main__':
    # Test the DataCollection class
    import random

    def random_array(length, low, high):
        return [random.randrange(low, high) for i in range(length)]

    NUM_MOVES = 10
    NUM_LABELS = 12
    EXP_LOCATION = os.path.join('data', 'test_exp')

    # Construct a list representing number of data points corresponding to each move.
    move_sizes = random_array(
        NUM_MOVES, segment.SEGMENT_SIZE * 4, segment.SEGMENT_SIZE * 8)
    move_sizes_sums = list(np.cumsum(move_sizes))
    move_starts = [0] + move_sizes_sums[:-1]
    num_data_points = move_sizes_sums[-1]
    print(move_starts)

    # Generate random sensor data
    sensors_data = [[sd.SensorDatum(random_array(sd.NUM_AXES, 0, 10), random_array(sd.NUM_AXES, 0, 10))
                     for i in range(NUM_SENSORS)
                     ]
                    for j in range(num_data_points)
                    ]
    labels = random_array(len(move_starts), 1, NUM_LABELS)

    # Process each data point
    data_collection = DataCollection(EXP_LOCATION)
    for i in range(num_data_points):
        if i in move_starts:
            data_collection.next_move()
        data_collection.process(sensors_data[i], random.random())

    data_collection.labels = labels

    assert data_collection.move_start_indices == move_starts

    # Save and reload
    data_collection.save()
    data_collection = DataCollection(EXP_LOCATION)
    data_collection.load()

    # Segment
    segments = data_collection.segment()
    print(labels)
    print(segments)
