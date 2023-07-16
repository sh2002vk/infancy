from .DataReader import DataReader
import numpy as np


def layer(function, a, weight, bias):
    newVec = a.dot(weight) + bias
    newVec = batchnorm(newVec)  # apply batch normalization
    a = function(newVec)
    return a


def batchnorm(a):
    """Center the data around 0, in batches"""
    mean = np.mean(a, axis=0)
    var = np.var(a, axis=0)
    return (a - mean) / np.sqrt(var + 1e-7)


def layernorm(a):
    """Normalize across features instead of batches"""
    mean = np.mean(a, axis=1, keepdims=True)
    var = np.var(a, axis=1, keepdims=True)
    return (a - mean) / np.sqrt(var + 1e-7)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    y = np.maximum(0, x)
    return y


def leakyReLU(x, alpha=0.01):
    """Alternate to regular ReLU"""
    return np.maximum(alpha * x, x)


def softmax(vector):
    e = np.exp(vector - np.max(vector))
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
        self.l1_bias = np.random.rand(1, 16)
        self.l2_weights = np.random.rand(16, 16) * np.sqrt(2. / 16)
        self.l2_bias = np.random.rand(1, 16)
        self.l3_weights = np.random.rand(16, 10) * np.sqrt(2. / 16)
        self.l3_bias = np.random.rand(1, 10)

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
        a = layer(leakyReLU, self.train_images, self.l1_weights, self.l1_bias)
        print(a[0])
        print("-----")
        a = layer(leakyReLU, a, self.l2_weights, self.l2_bias)
        print(a[0])
        print("-----")
        a = layer(softmax, a, self.l3_weights, self.l3_bias)
        print(a[0])
        print("-----")
        return a



