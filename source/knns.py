import pandas as pd
import os
import random
import math
import time
import operator
import numpy as np
from random import shuffle
from scipy.io import arff


class KNN:

	def __init__(self, dataset):
		self.dataset = dataset
		self.folds = []
		self.load_dataset(dataset)

	def load_dataset(self, db_name):
		db_path = os.path.join(
			os.path.abspath(os.path.curdir), 'databases')

		db = arff.loadarff(os.path.join(db_path, db_name))
		df = pd.DataFrame(db[0])
		values = df.values
		values = values.tolist()
		for value in values:
			b = value[-1].decode('utf-8')
			if b == 'false' or b == 'no':
				value[-1] = False
			else:
				value[-1] = True

		self.folds = np.array(values)

	def get_neighbours(self, db_train, test, k, distance, weight):
		distances = []
		args = len(test) - 2

		for x in db_train:
			dist = distance(test, x, args)
			w = 1/(dist + 0.1)
			if weight:
				distances.append((x, w))
			else:
				distances.append((x, dist))
		distances.sort(key=operator.itemgetter(1))

		neighbours = []
		for x in range(k):
			neighbours.append(distances[x][0])

		return neighbours

	def predict(self, neighbours):
		class1 = 0
		class2 = 0

		for n in neighbours:
			if not n[-1]:
				class1 += 1
			else:
				class2 += 1

		if class1 > class2:
			return 0

		else:
			return 1
