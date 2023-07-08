import numpy as np
import struct


class DataReader:
    def __init__(self, train_images_path, train_labels_path):
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path

    def load_and_read_data(self):
        with open(self.train_labels_path, 'rb') as file:
            file.read(8)
            labels = np.frombuffer(file.read(), dtype=np.uint8)

        with open(self.train_images_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_data = np.frombuffer(file.read(), dtype=np.uint8)

        images = []
        for i in range(size):
            images.append([0]*rows*cols)   # List of images, currently initialized with all 0s
        for i in range(size):
            img = np.array(image_data[i*rows*cols:(i+1)*rows*cols])
            img.reshape(28, 28)     # Will be 28*28=784 2D array
            images[i][:] = img

        return np.array(images), np.array(labels)
