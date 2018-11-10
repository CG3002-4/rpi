import sys
import pickle
import numpy as np
from segment import Segment, SEGMENT_SIZE, SEGMENT_OVERLAP
from preprocess import preprocess_segment
from feature_extraction import extract_features_over_segment
import clientconnect
from clientconnect import recv_data, send_data, create_socket
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
            self.portion_of_segment_complete = int(SEGMENT_SIZE * SEGMENT_OVERLAP)

            return self.make_prediction()

    def make_prediction(self):
        segment = Segment(self.data, None)
        segment = preprocess_segment(segment)
        features = extract_features_over_segment(segment)
        features = np.nan_to_num(features)

        return self.model.predict_proba(features.reshape(1, -1))[0]

    def clear_data(self):
        self.data = np.empty((0, 12))
        self.portion_of_segment_complete = 0


NUM_PREDS_TO_KEEP = 3
NUM_NEUTRAL_THROW = 2
NUM_MOVES = 2
PREDICTION_THRESHOLD = 0.7
TIME_TO_DISCARD = 1.5
MOVES = {
    0: 'neutral',
    1: 'wipers',
    2: 'number7',
    3: 'chicken',
    4: 'sidestep',
    5: 'turnclap',
    6: 'number6',
    7: 'salute',
    8: 'mermaid',
    9: 'swing',
    10: 'cowboy',
    11: 'logout'
}


class Predictor:
    def __init__(self, model_file):
        self.segment_predictor = SegmentPredictor(model_file)
        self.predictions = np.empty((0, NUM_MOVES))
        self.prevPredictionTime = time.time()

    def process(self, sensors_datum):
        if time.time() - self.prevPredictionTime < TIME_TO_DISCARD:
            return

        segment_prediction = self.segment_predictor.process(sensors_datum)

        if segment_prediction is not None:
            assert len(segment_prediction) == NUM_MOVES

            if len(self.predictions) < NUM_PREDS_TO_KEEP:
                self.predictions = np.vstack(
                    [self.predictions, segment_prediction])
            else:
                self.predictions = np.vstack(
                    [self.predictions[1:], segment_prediction])

            if len(self.predictions) == NUM_NEUTRAL_THROW:
                self.check_neutral()

            if len(self.predictions) == NUM_PREDS_TO_KEEP:
                return self.make_prediction()

    def check_neutral(self):
        probabilities = np.prod(self.predictions, axis=0)
        normalized_probs = probabilities / np.sum(probabilities)
        prediction = np.argmax(normalized_probs)
        
        if max(normalized_probs) > PREDICTION_THRESHOLD and prediction == 0:
            self.predictions = np.empty((0, NUM_MOVES))
            self.segment_predictor.clear_data()
            print('Cleared neutral!')

    def make_prediction(self):
        probabilities = np.prod(self.predictions, axis=0)
        normalized_probs = probabilities / np.sum(probabilities)
        print('Probabilities: ' + str(normalized_probs))

        if max(normalized_probs) > PREDICTION_THRESHOLD:
            prediction = np.argmax(normalized_probs)
            self.prevPredictionTime = time.time()
            self.predictions = np.empty((0, NUM_MOVES))
            self.segment_predictor.clear_data()

            return prediction


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.seterr(divide='ignore', invalid='ignore')

    #socket = create_socket(sys.argv[2], int(sys.argv[3]))
    predictor = Predictor(model_file=sys.argv[1])

    print('Loaded model')

    energy = 0
    prev_time = None

    # with open('realtime_' + str(int(time.time())) + '.log', 'w') as log_file:
    for unpacked_data in recv_data():
        # log_file.write(str(list(unpacked_data)) + '\n')

        voltage = unpacked_data[12]
        current = unpacked_data[13]
        power = voltage * current

        # print(unpacked_data)
        if prev_time is not None:
            energy += power * (time.time() - prev_time)
            # print(time.time() - prev_time)
        prev_time = time.time()

        prediction = predictor.process(unpacked_data[:12])
        if prediction is not None:
            print('Prediction: ' + MOVES[prediction])
            # send_data(socket, prediction, voltage, current, power, energy)
