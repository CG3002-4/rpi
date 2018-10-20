import sys
import pickle
import numpy as np
from recv_data import recv_data
from data_collection import NUM_SENSORS
from sensor_data import SensorDatum, sensor_datums_to_sensor_data
from segment import Segment, SEGMENT_SIZE, SEGMENT_OVERLAP
from preprocess import preprocess_segment
from feature_extraction import extract_features_over_segment
import time

class SegmentPredictor:
    def __init__(self, model_file):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        self.data = np.empty((0, 12))
        self.portion_of_segment_complete = 0

    def process(self, sensors_datum):
        assert len(sensors_datum) == 12

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
        # TODO: Takes too much time
        segment = Segment(self.data, None)
        segment = preprocess_segment(segment)
        features = extract_features_over_segment(segment)
        features = np.nan_to_num(features)

        return self.model.predict_proba(features.reshape(1, -1))[0]


NUM_PREDS_TO_KEEP = 2
NUM_MOVES = 6
PREDICTION_THRESHOLD = 0.7


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
            print(np.argmax(normalized_probs))
            self.predictions = np.empty((0, NUM_MOVES))
            return np.argmax(normalized_probs)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.seterr(divide='ignore', invalid='ignore') 

    predictor = Predictor(model_file=sys.argv[1] + '.pb')

    print('Loaded model')

    for unpacked_data in recv_data():
        # Need to call server comm code here
        predictor.process(unpacked_data[:12])

