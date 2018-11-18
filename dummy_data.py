import numpy as np
import time

DATA_LEN = 1000

def recv_data():
    for i in range(DATA_LEN):
        time.sleep(0.016)
        yield np.random.rand(14)
