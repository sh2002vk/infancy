from .DataReader import DataReader
import numpy as np


def layer(function, a, weight, bias):
    newVec = a.dot(weight)
    a = function(newVec + bias)
    return a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    y = np.maximum(0, x)
    return y


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


class NeuralNet:
    def __init__(self, train, test):
        # QUESTION: Would it be a better design decision to update DatReader to take in train and test at the same time?
        tempReader = DataReader(train)
        self.train_images, self.train_labels = tempReader.load_and_read_data()
        self.train_images = self.train_images / 255

        tempReader = DataReader(test)
        self.test_images, self.test_labels = tempReader.load_and_read_data()

        # NETWORK CONFIGURATION Input Layer (784) -> Hidden Layer (16) -> Hidden Layer (16) -> Output Layer (10)
        # Consider Glorot initialization or He initialization

        self.l1_weights = np.random.rand(784, 16) * np.sqrt(2. / 784)
        self.l1_bias = np.zeros(16)
        self.l2_weights = np.random.rand(16, 16) * np.sqrt(2. / 16)
        self.l2_bias = np.zeros(16)
        self.l3_weights = np.random.rand(10, 10) * np.sqrt(2. / 16)
        self.l3_bias = np.zeros(16)

    def forward_propogate(self):
        """
        General Calculation: sigmoid((ACTIVATION_MATRIX * WEIGHT_MATRIX) + BIAS_VECTOR)
        Where A[0] is the input layer, n is the count of images, M is the batch size
        Input -> HL1:
            A[1][n] = sigmoid(((M, 784) * (784, 16)) + (1, 16))
        HL1 -> HL2:
            A[2][n] = sigmoid(((1, 16) * (16, 16)) + (1, 16))
        HL2 -> Output:
            Output = softmax(((1, 16) * (16, 10)) + (1, 10))
        """
        a = layer(ReLU, self.train_images, self.l1_weights, self.l1_bias)
        a = layer(ReLU, a, self.l2_weights, self.l2_bias)
        a = layer(softmax, a, self.l3_weights, self.l3_bias)
        return a



