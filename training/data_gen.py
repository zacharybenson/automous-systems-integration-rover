import glob
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self,
                 DEFAULT_DATA_PATH,
                 data,
                 use_sequence,
                 sequence_size,
                 input_size,
                 batch_size):

        self.use_sequence = use_sequence
        self.DEFAULT_DATA_PATH = DEFAULT_DATA_PATH
        self.data = data
        self.n = len(data)
        self.sequence_size = sequence_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.batches = ''
        self.sequences = ''

    # This will create batch based on sequences:

    def create_batches(self, sequences):
        batches = []
        num_of_batches = len(sequences) // self.batch_size
        for i in range(num_of_batches):
            batch = sequences[i:i + self.batch_size]
            batches.append(batch)

        self.batches = batches
        return batches

    # This will get sequences of images.
    def create_sequences(self, data):
        data = data.values.tolist()
        sequences = []
        for i in range(len(data)):
            if i == (len(data) - self.sequence_size):
                break
            sequence = [data[i + j] for j in range(self.sequence_size)]
            sequences.append(sequence)
        self.sequences = sequences

        return sequences

    def apply_mask(self, image):
        image = np.asanyarray(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sensitivity = 85
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        white_range = cv2.bitwise_and(image, image, mask=mask)

        return white_range

    def crop_image(self, image):
        width, height = image.size

        # Setting the points for cropped image
        left = width / 3
        right = width

        top = height / 2
        bottom = height

        # Cropped image of above dimension
        # (It will not change original image)
        image_cropped = image.crop((left, top, right, bottom))

        return image_cropped

    def __get_input(self, samples, target_size):
        # for sample in samples:
        for sample in samples:
            image = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
            return image / 255

    def __get_output(self, samples):
        path = samples[0]
        y = path.split('/')[6]
        y0 = float(y.split('_')[1])
        y1 = float(y.split('_')[2])

        return (y0, y1)

    def __get_data(self, batches, index):
        # Generates data containing batch_size samples
        if index <= self.__len__():
            batch = batches[index]
            X_batch = np.asarray([self.__get_input(x, self.input_size) for x in batch])
            y_batch = np.asarray([self.__get_output(y) for y in batch])
        return X_batch, y_batch

    def __getitem__(self, index):

        if self.use_sequence:
            sequences = self.create_sequences(self.data)
            sequences = list(np.concatenate(sequences))
            batches = self.create_batches(sequences)
            X, y = self.__get_data(batches, index)

        else:
            batches = self.create_batches(self.data.values.tolist())
            X, y = self.__get_data(batches, index)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.batches)


def create_list_of_data(path, ends_with):
    cols = ["file_name"]
    files_processed = []
    ends_with = ends_with + ".png"

    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

    for folder in subfolders:
        path = os.path.join(folder, "*.png")
        files = sorted(glob.glob(path))
        for file in files:
            # only list main RGB images
            if file.endswith(ends_with):
                files_processed.append(file)

    df = pd.DataFrame(files_processed)
    df.columns = cols

    return df
