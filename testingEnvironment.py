from Classes.DataReader import DataReader
import matplotlib.pyplot as plt

reader = DataReader('archive/train-images.idx3-ubyte', 'archive/train-labels.idx1-ubyte')
images, labels = reader.load_and_read_data()

