import pickle
import segment
import numpy as np


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
    def __init__(self, filename):
        self.sensors_data = []
        self.move_start_indices = []
        self.filename = filename

    def process(self, sensors_datum):
        """Takes in a list of sensor_data.SensorDatum representing one data
        point for each sensor."""
        self.sensors_data.append(sensors_datum)

    def next_move(self):
        self.move_start_indices.append(len(self.sensors_data))

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        with open(self.filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

    def segment(self, labels):
        """Given labels for each of the moves, segment the data.

        Note that this function simply segments the data according to the
        segment size and overlap values in segment.py.

        It then labels each segment by checking which label accounts for
        the majority of the segment.
        """
        assert len(labels) == len(self.move_start_indices)

        num_data_points = len(self.sensors_data)
        sensors_data = np.array(self.sensors_data)

        # To make segmenting a little easier
        self.move_start_indices.append(num_data_points)

        segment_start = 0
        curr_move_idx = 0
        segments = []
        while segment_start + segment.SEGMENT_SIZE <= num_data_points:
            # Compute what part of this segment is labelled by the current
            # index.
            portion_in_curr_move = self.move_start_indices[curr_move_idx + 1] - segment_start
            if portion_in_curr_move < segment.SEGMENT_SIZE // 2:
                # label with next move
                curr_move_idx += 1

            segments.append(segment.Segment(sensors_data[segment_start: segment_start + segment.SEGMENT_SIZE], labels[curr_move_idx]))

            segment_start += segment.SEGMENT_OFFSET

        # Undo mutation made to this array at the start
        self.move_start_indices.pop()

        return segments


if __name__ == '__main__':
    # Test the DataCollection class
    import numpy as np
    import random
    import sensor_data

    def random_array(length, low, high):
        return [random.randrange(low, high) for i in range(length)]

    NUM_MOVES = 10
    NUM_LABELS = 12
    NUM_SENSORS = 2
    NUM_AXES = 3
    DATA_FILE = 'collection_test.pb'

    # Construct a list representing number of data points corresponding to each move.
    move_sizes = random_array(NUM_MOVES, segment.SEGMENT_SIZE * 4, segment.SEGMENT_SIZE * 8)
    move_sizes_sums = list(np.cumsum(move_sizes))
    move_starts = [0] + move_sizes_sums[:-1]
    num_data_points = move_sizes_sums[-1]
    print(move_starts)

    # Generate random sensor data
    sensors_data = [[sensor_data.SensorDatum(random_array(NUM_AXES, 0, 10), random_array(NUM_AXES, 0, 10))
                     for i in range(NUM_SENSORS)
                     ]
                    for j in range(num_data_points)
                    ]
    labels = random_array(len(move_starts), 1, NUM_LABELS)

    # Process each data point
    data_collection = DataCollection(DATA_FILE)
    for i in range(num_data_points):
        if i in move_starts:
            data_collection.next_move()
        data_collection.process(sensors_data[i])

    assert data_collection.move_start_indices == move_starts

    # Save and reload
    data_collection.save()
    data_collection = DataCollection(DATA_FILE)
    data_collection.load()

    # Segment
    segments = data_collection.segment(labels)
    print(labels)
    print(segments)
