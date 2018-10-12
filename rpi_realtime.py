import sys
import pickle
import numpy as np
from recv_data import recv_data
from data_collection import NUM_SENSORS
from sensor_data import SensorDatum, sensor_datums_to_sensor_data
from segment import Segment, SEGMENT_SIZE, SEGMENT_OVERLAP
from preprocess import preprocess_segment
from feature_extraction import extract_features_over_segment
from pipeline import NOISE_FILTERS, FEATURE_EXTRACTORS


class Predictor:
    def __init__(self, model_file):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        self.data = np.empty((0, NUM_SENSORS))
        self.portion_of_segment_complete = 0

    def process(self, sensors_datum):
        assert len(sensors_datum) == NUM_SENSORS

        if len(self.data) < SEGMENT_SIZE:
            self.data = np.vstack([self.data, sensors_datum])
        else:
            # len(self.data) has to be equal to segment size
            self.data = np.vstack([self.data[1:], sensors_datum])

        self.portion_of_segment_complete += 1

        if self.portion_of_segment_complete == SEGMENT_SIZE:
            self.make_prediction()

            # Up to segment_offset is already contained in the data
            self.portion_of_segment_complete = SEGMENT_SIZE * SEGMENT_OVERLAP

    def make_prediction(self):
        sensors_data = np.apply_along_axis(func1d=sensor_datums_to_sensor_data, arr=self.data, axis=0)
        segment = Segment(sensors_data, None)
        segment = preprocess_segment(segment, NOISE_FILTERS)
        features = extract_features_over_segment(segment, FEATURE_EXTRACTORS)
        print(self.model.predict(features))


if __name__ == '__main__':
    predictor = Predictor(model_file=sys.argv[1] + '.pb')

    for unpacked_data in recv_data():
        sensor1_datum = SensorDatum(
            unpacked_data[0:3], unpacked_data[3:6])
        sensor2_datum = SensorDatum(
            unpacked_data[6:9], unpacked_data[9:12])

        predictor.process([sensor1_datum, sensor2_datum])
