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


class SegmentPredictor:
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
            # Up to segment_overlap is already contained in the data
            self.portion_of_segment_complete = SEGMENT_SIZE * SEGMENT_OVERLAP

            return self.make_prediction()

    def make_prediction(self):
        sensors_data = np.apply_along_axis(func1d=sensor_datums_to_sensor_data, arr=self.data, axis=0)
        segment = Segment(sensors_data, None)
        segment = preprocess_segment(segment, NOISE_FILTERS)
        features = extract_features_over_segment(segment, FEATURE_EXTRACTORS)
        return self.model.predict_proba(features)[0]


NUM_PREDS_TO_KEEP = 15
NUM_MOVES = 5
PREDICTION_THRESHOLD = 0.8


class Predictor:
    def __init__(self, model_file):
        self.segment_predictor = SegmentPredictor(model_file)
        self.predictions = np.empty((0, NUM_MOVES))

    def process(self, sensors_datum):
        segment_prediction = self.segment_predictor.process(sensors_datum)

        if segment_prediction is not None:
            assert len(segment_prediction) == NUM_MOVES

            if len(self.predictions) < NUM_PREDS_TO_KEEP:
                self.predictions = np.vstack([self.predictions, segment_prediction])
            else:
                self.predictions = np.vstack([self.predictions[1:], segment_prediction])

            if len(self.predictions) == NUM_PREDS_TO_KEEP:
                return self.make_prediction()

    def make_prediction(self):
        probabilities = np.prod(self.predictions, axis=0)
        normalized_probs = probabilities / np.sum(probabilities)
        print('Probabilities: ' + str(normalized_probs))

        if max(normalized_probs) > PREDICTION_THRESHOLD:
            print(max(normalized_probs))
            return np.argmax(normalized_probs) + 1


if __name__ == '__main__':
    import random

    def random_array(length, low, diff):
        return [random.random() * diff + low for i in range(length)]

    np.set_printoptions(suppress=True)

    predictor = Predictor(model_file=sys.argv[1] + '.pb')

    for unpacked_data in [random_array(12, -5, 5) for j in range(200)]:
        sensor1_datum = SensorDatum(
            unpacked_data[0:3], unpacked_data[3:6])
        sensor2_datum = SensorDatum(
            unpacked_data[6:9], unpacked_data[9:12])

        # Need to call server comm code here
        predictor.process([sensor1_datum, sensor2_datum])
