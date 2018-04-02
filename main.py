import pandas as pd
import os
import random
import math
import time
import operator
import numpy as np
from random import shuffle
from sklearn.model_selection import KFold
from scipy.io import arff

# kc1_train, kc1_test = load_dataset('kc1.arff')
# kc2_train, kc2_test = load_dataset('kc2.arff')


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

	def cross_validation(self, hm_neighbours):
		ac = []
		kf = KFold(n_splits=10, shuffle=True)
		for k in hm_neighbours: 
			localac = []
			for train_index, test_index in kf.split(self.folds):
				db_train = self.folds[train_index]
				db_train = db_train.tolist()
				db_test = self.folds[test_index]
				db_test = db_test.tolist()
				localac.append(0)				
				for test in db_test:
					neighbours = self.get_neighbours(
						db_train, test, k, self.euclidian_distance, True)
					result = self.get_result(neighbours)
					if (int(test[-1])==result):
						localac[-1] += 1
				localac[-1] = localac[-1]/len(db_test)
			ac.append(0)
			for lac in localac:
				ac[-1] += lac
			ac[-1] /= len(localac)
			print('Accuracy for k={} :'.format(k), ac[-1])

		return ac

	def euclidian_distance(self, in1, in2, hm_args):
		dist = 0
		for x in range(hm_args):
			dist += pow((in1[x] - in2[x]), 2)

		return math.sqrt(dist)

	def get_neighbours(self, db_train, test, k, distance, weight):
		distances = []
		args = len(test) - 2

		for x in db_train:
			dist = distance(test, x, args)
			distances.append((x, dist))
		distances.sort(key=operator.itemgetter(1))

		neighbours = []
		for x in range(k):
			neighbours.append(distances[x][0])

		return neighbours

	def get_result(self, neighbours):
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

knn = KNN('kc2.arff')
knn.cross_validation(hm_neighbours=[1,2,3,5,7,9,11,13,15])
