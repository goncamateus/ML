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


if __name__ == '__main__':
	
	dataset = load_dataset('kc1.arff')

	lvqs = ['Nothing', lvq1, lvq21, lvq3]
	scores = np.zeros(shape=(4,1))

	# fig, ax = plt.subplots()
	# image, = ax.plot()
	# plt.ylabel('Accuracy')
	# plt.xlabel('Neighbours')
	# plt.draw()
	
	for i, lvq in enumerate(lvqs):
		
		data = dataset
		np.random.shuffle(data)		
		if i!=0:
			before = time.time()		
			prototypes = lvq(data, 10)
			data = np.concatenate((data, prototypes), axis=0)
			ts = time.time() - before

		np.random.shuffle(data)		

		data_train = data[:-int(len(data)*0.67)]
		data_test = data[int(len(data)*0.67):]

		for x in data_test:
			knn = KNN()
			neighbours = knn.get_neighbours(data_train, x, 3)
			if knn.predict(neighbours) == x[-1]:
				scores[i] += 1

		scores[i] = scores[i]/len(data_test)
		print(scores[i])

		# image.set_data(scores[i])
		# plt.suptitle(lvq.__name__, fontsize=14, fontweight='bold')
		# plt.title('Timestamp: {} seconds'.format(ts))
		# plt.draw()
		# fig.savefig("{}.png".format(lvq.__name__))
