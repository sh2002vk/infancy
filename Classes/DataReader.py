import numpy as np
import pandas as pd


class DataReader:
    def __init__(self, data):
        self.path = data

    def load_and_read_data(self):
        data = pd.read_csv(self.path)
        data = np.array(data)
        labels = data[:, 0]
        images = data[:, 1:]
        return images, labels
