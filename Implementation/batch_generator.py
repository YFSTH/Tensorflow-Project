import os
import pickle
import numpy as np


class MNISTCollage:
    """ Helper class for initializing MNIST Collage data set
    """
    def __init__(self, directory):
        files = os.listdir(directory)

        for filename in files:
            size = filename.split("_")
            if "train_collages" in filename:
                self.train_data = self.load_data(os.path.join(directory, filename))
                self.train_data = np.reshape(self.train_data[0], (int(size[0]), int(size[1]), int(size[1]), 1))
                self.train_data = np.repeat(self.train_data, 3, axis=3)
            if "train_targets" in filename:
                self.train_labels = self.load_data(os.path.join(directory, filename))
            if "valid_collages" in filename:
                self.valid_data = self.load_data(os.path.join(directory, filename))
                self.valid_data = np.reshape(self.valid_data[0], (int(size[0]), int(size[1]), int(size[1]), 1))
                self.valid_data = np.repeat(self.valid_data, 3, axis=3)
            if "valid_targets" in filename:
                self.valid_labels = self.load_data(os.path.join(directory, filename))
            if "test_collages" in filename:
                self.test_data = self.load_data(os.path.join(directory, filename))
                self.test_data = np.reshape(self.test_data[0], (int(size[0]), int(size[1]), int(size[1]), 1))
                self.test_data = np.repeat(self.test_data, 3, axis=3)
            if "test_targets" in filename:
                self.test_labels = self.load_data(os.path.join(directory, filename))

    def load_data(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def get_batch(self, data, labels, batch_size):
        for i in range(len(data) // batch_size):
            first = i * batch_size
            last = first + batch_size
            yield np.squeeze(data[first:last]), labels[first:last]
