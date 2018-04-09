import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from scipy.io import arff
from source.lvqs import lvq1, lvq21, lvq3
from source.knn import KNN


def load_dataset(db_name):
	db_path = os.path.join(
			os.path.abspath(os.path.pardir), 'databases')
	if(db_name.endswith('.arff')):
		db = arff.loadarff(os.path.join(db_path, db_name))
		df = pd.DataFrame(db[0])
	else:
		df = pd.read_csv(os.path.join(db_path, db_name))
	values = df.get_values()
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

def no_generator():
	pass


if __name__ == '__main__':
	

	dataset = load_dataset('kc1.arff')

	lvqs = [no_generator, lvq1, lvq21, lvq3]
	scores = np.zeros(shape=(4,2))
	hw_many = 10

	for i, lvq in enumerate(lvqs):
		
		data = dataset
		np.random.shuffle(data)
		before = time.time()		
		if i!=0:			
			prototypes = lvq(data, hw_many, weight=False)
			ts = time.time() - before
			for j,k in enumerate([1,3]):
				for shu in range(10):
					np.random.shuffle(data)
				np.random.shuffle(prototypes)
				data_train = prototypes
				data_test = data[:500]
				in_score = 0
				for x in data_test:
					knn = KNN()
					neighbours = knn.get_neighbours(data_train, x, k, weight=False)
					if knn.predict(neighbours) == x[-1]:
						in_score += 1
				scores[i][j] = in_score/len(data_test)
		else:
			for j,k in enumerate([1,3]):
				for shu in range(10):
					np.random.shuffle(data)
				data_train = data[:-int(len(data)*0.67)]
				data_test = data[int(len(data)*0.67):]
				in_score = 0
				for x in data_test:
					knn = KNN()
					neighbours = knn.get_neighbours(data_train, x, k, weight=False)
					if knn.predict(neighbours) == x[-1]:
						in_score += 1
				scores[i][j] = in_score/len(data_test)

		print(scores[i])

	plt.plot(scores)
	plt.suptitle('LVQs', fontsize=14, fontweight='bold')
	plt.title('Comparisson between methods')
	plt.draw()
	plt.savefig("lvqs.png")

