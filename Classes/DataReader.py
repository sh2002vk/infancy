import numpy as np
import pandas as pd


class DataReader:
    def __init__(self, data):
        self.path = data

    def load_and_read_data(self):
        data = pd.read_csv(self.path)
        data = np.array(data)
        np.random.shuffle(data)
        labels = data[:, 0]
        images = data[:, 1:]
        # num_classes = np.max(labels) + 1  # Assumes classes are 0-indexed and continuous
        # labels_one_hot = np.eye(num_classes)[labels]  # One-hot encode labels
        # print(labels_one_hot[0])
        print(labels)
        return images, labels
