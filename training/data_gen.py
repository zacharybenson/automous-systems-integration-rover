import os
from itertools import chain

import tensorflow as tf
import pandas as pd
import numpy as np

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

	# This will create batch based on sequences:


	def create_batches(self,sequences):
		batches = []
		num_of_batches = len(sequences) // self.batch_size
		for i in range(num_of_batches):
			batch = sequences[i:i+self.batch_size]
			batches.append(batch)


		return batches



	# This will get sequences of images.
	def create_sequences(self,data):
		data = data.values.tolist()
		sequences = []
		for i in range(len(data)):
			if i == (len(data) - self.sequence_size):
				break
			sequence = [data[i + j] for j in range(self.sequence_size)]
			sequences.append(sequence)

		return sequences

	def __get_input(self, samples, target_size):
		# for sample in samples:
		image = tf.keras.preprocessing.image.load_img(self.DEFAULT_DATA_PATH + samples[0])
		image = tf.image.rgb_to_grayscale(image)
		image_arr = tf.keras.preprocessing.image.img_to_array(image)
		image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
		return image_arr/255


	def __get_output(self, samples,index):
		path = samples[0]
		y = float(path.split('_')[index])

		return y

	def __get_data(self, batches, index):
		# Generates data containing batch_size samples
		if index <= self.__len__():

			batch = batches[index]
			X_batch = np.asarray([self.__get_input(x, self.input_size) for x in batch])
			y0_batch = np.asarray([self.__get_output(y,1) for y in batch])
			y1_batch = np.asarray([self.__get_output(y,2) for y in batch])


		return X_batch,y0_batch, y1_batch

	def __getitem__(self, index):

		if self.use_sequence:
			sequences = self.create_sequences(self.data)
			sequences = list(np.concatenate(sequences))
			batches = self.create_batches(sequences)
			X, y0, y1 = self.__get_data(batches, index)


		else:
			batches = self.create_batches(self.data.values.tolist())
			X, y0, y1 = self.__get_data(batches,index)
		return X, y0, y1

	def __len__(self):
		return self.n // self.batch_size


#Implement Series processing
#Implement Inference



def create_list_of_data(path,ends_with):
	cols = ["index", "file_name"]
	files = []

	for file in os.listdir(path):
		if file.endswith(ends_with + ".png"):
			index = int(file.split('_')[0].lstrip("0"))
			files.append([index, file])
	df = pd.DataFrame(files)
	df.columns = cols
	df.set_index('index', inplace=True)
	df = df.sort_index(ascending=True)

	return df