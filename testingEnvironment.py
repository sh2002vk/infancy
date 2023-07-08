from Classes.NeuralNet import NeuralNet
from Classes.DataReader import DataReader

net = NeuralNet('mnist_train.csv',
                'mnist_test.csv')
net.forward_propogate()
