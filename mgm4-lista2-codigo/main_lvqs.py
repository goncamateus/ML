import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from source.lvqs import lvq1, lvq21, lvq3
from prototype import generation


def load_dataset(db_name):
	db_path = os.path.join(
			os.path.abspath(os.path.pardir), 'databases')
	if(db_name.endswith('.arff')):
		db = arff.loadarff(os.path.join(db_path, db_name))
		df = pd.DataFrame(db[0])
	else:
		df = pd.read_csv(os.path.join(db_path, db_name))

	df_norm = (df.iloc[:,:-1] - df.iloc[:,:-1].mean()) / (df.iloc[:,:-1].max() - df.iloc[:,:-1].min())
	values = df.get_values()
	values[:,:-1] = df_norm

	for value in values:
		try:
			b = value[-1].decode('utf-8')
		except AttributeError:
			b = value[-1]

		if (b == 'false'):
			value[-1] = 0
		else:
			value[-1] = 1


	return values


if __name__ == '__main__':

	dataset = load_dataset('jm1.arff')

	lvqs = [lvq1, lvq21, lvq3]
	scores = np.zeros(shape=(4,2))
	hw_many = 10
	protos = generation(dataset, hw_many)

	for i, lvq in enumerate(lvqs):

		data = dataset
		for shu in range(10):
			np.random.shuffle(data)

		before = time.time()					
		prototypes = lvq(data, protos, hw_many, weight=False)
		ts = time.time() - before

		for j,k in enumerate([1,3]):

			for shu in range(10):
				np.random.shuffle(data)

			data_train = prototypes
			data_test = data
			y_test = data_test[:,-1]

			knn = KNeighborsClassifier(n_neighbors=k)
			knn.fit(data_train[:, :-1], data_train[:,-1])
			pred = knn.predict(data_test[:, :-1])
			ok = 0
			for t in range(len(pred)):
				if pred[t]  == y_test[t]:
					ok += 1
			print(ok/len(pred))
