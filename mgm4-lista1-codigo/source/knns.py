import pandas as pd
import os
import random
import math
import operator
import numpy as np
from random import shuffle
from scipy.io import arff
from source.distances import euclidian_distance, vdm


class KNN:

	def __init__(self, dataset):
		self.dataset = dataset
		self.folds = []
		self.hm_vars = 0
		self.load_dataset(dataset)

	def load_dataset(self, db_name):
		db_path = os.path.join(
			os.path.abspath(os.path.pardir), 'databases')
		if(self.dataset.endswith('.arff')):
			db = arff.loadarff(os.path.join(db_path, db_name))
			df = pd.DataFrame(db[0])
		else:
			df = pd.read_csv(os.path.join(db_path, db_name))

		values = df.values
		values = values.tolist()
		self.hm_vars = len(values[0])
		for value in values:

			try:
				b = value[-1].decode('utf-8')
			except AttributeError:
				b = value[-1]

			if (b == 'false' or b == 'no' or \
				b == 'nowin' or b == 'negative'\
				or b == '2' or b == 'F'):

				value[-1] = 0
			else:
				value[-1] = 1
		self.folds = np.array(values)

	def get_neighbours(self, db_train, test, k, distance=euclidian_distance, weight=False):
		distances = []
		args = len(test) - 1

		if(distance == euclidian_distance):
			for x in db_train:
				dist = distance(test, x, args)
				w = 1/(dist + 0.1)
				if weight:
					distances.append((x, dist*w))
				else:
					distances.append((x, dist))

		else:
			distances = distance(test, args, db_train, weight)
		distances.sort(key=operator.itemgetter(1))
		neighbours = []


		for x in range(k):
			neighbours.append(distances[x][0])
		#print('Neighbours:\n',neighbours)
		return neighbours

	def predict(self, neighbours):
		class1 = 0
		class2 = 0

		for n in neighbours:
			if n[-1] == 0:
				class1 += 1
			else:
				class2 += 1
		if class1 > class2:
			return 0

		else:
			return 1
