from .DataReader import DataReader
import matplotlib.pyplot as plt
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


def derivative_leakyReLU(x, alpha=0.01):
    output = np.ones_like(x)
    output[x < 0] = alpha
    return output


def softmax(vector):
    e = np.exp(vector - np.max(vector))
    return e / e.sum()


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y


class NeuralNet:

    def __init__(self, train, test):
        # QUESTION: Would it be a better design decision to update DatReader to take in train and test at the same time?
        tempReader = DataReader(train)
        self.train_images, self.train_labels = tempReader.get_train_test_split()
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
        self.a0, self.a1, self.a2 = None, None, None    # These activations will be used to improve weights in backprop

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
        self.a0 = layer(leakyReLU, self.train_images, self.l1_weights, self.l1_bias)
        # print(self.a0[0])
        # print("-----")
        self.a1 = layer(leakyReLU, self.a0, self.l2_weights, self.l2_bias)
        # print(self.a1[0])
        # print("-----")
        self.a2 = layer(softmax, self.a1, self.l3_weights, self.l3_bias)
        # print(self.a2[0])
        # print("-----")
        return self.a2

    def back_propogate(self):
        train_labels = one_hot(self.train_labels)
        num_samples = len(train_labels)
        l3_error = self.a2 - train_labels

        temp = self.l3_weights
        l2_error = l3_error.dot(temp.T) * derivative_leakyReLU(self.a1)

        temp = self.l2_weights
        l1_error = l2_error.dot(temp.T) * derivative_leakyReLU(self.a0)

        updated_bias_3 = np.sum(l3_error, axis=0, keepdims=True)
        updated_weights_3 = self.a1.T.dot(l3_error)
        updated_bias_2 = np.sum(l2_error, axis=0, keepdims=True)
        updated_weights_2 = self.a0.T.dot(l2_error)
        updated_bias_1 = np.sum(l1_error, axis=0, keepdims=True)
        updated_weights_1 = self.train_images.T.dot(l1_error)

        learning_rate = 0.08
        self.l1_weights -= learning_rate * updated_weights_1 / num_samples
        self.l1_bias -= learning_rate * updated_bias_1 / num_samples
        self.l2_weights -= learning_rate * updated_weights_2 / num_samples
        self.l2_bias -= learning_rate * updated_bias_2 / num_samples
        self.l3_weights -= learning_rate * updated_weights_3 / num_samples
        self.l3_bias -= learning_rate * updated_bias_3 / num_samples

    def get_predictions(self):
        return np.argmax(self.a2, 0)

    def get_accuracy(self, predictions, labels):
        return np.sum(predictions == labels) / labels.size

    def cross_entropy_loss(self, predictions):
        """ Entropy is the level of uncertainty in the possible output of given inputs.
        The greater the entropy, the higher the uncertainty.

        This function is also referred to as "logarithmic loss"
        L(CE) = -(SUMMATION(t(i) * log(p(i)))) where t(i) is truth, p(i) is output of softmax and i is the class

        :return:
        """
        train_labels = one_hot(self.train_labels)
        num_samples = len(train_labels)
        return -np.sum(train_labels * np.log(predictions)) / num_samples

    def train(self, epochs):
        losses = []
        for i in range(epochs):
            predictions = self.forward_propogate()
            losses.append(self.cross_entropy_loss(predictions))
            self.back_propogate()
        self.performance_report(losses)

    def performance_report(self, losses):
        plt.figure()
        plt.plot(losses)
        plt.title("Loss over time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()



