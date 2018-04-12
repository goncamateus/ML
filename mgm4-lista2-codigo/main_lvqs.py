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

		if (b in ['false','no']):
			value[-1] = 0
		else:
			value[-1] = 1


	return values


if __name__ == '__main__':

	dataset = load_dataset('jm1.arff')

	lvqs = [lvq1, lvq21, lvq3]
	scores = np.zeros(shape=(4,2))
	hw_many = 100
	protos = generation(dataset, hw_many)

	classes = protos[:,-1]
	falses = (classes == 0).sum()
	x = np.arange(2)
	heigh = [falses, len(classes)-falses]

	bar = plt.figure(1)
	plt.bar(x, heigh)
	plt.xticks(x,('False', 'True'))
	plt.ylim((0, len(classes)))
	plt.title('Data balance {} prototypes'.format(hw_many))
	plt.suptitle('Prototypes')
	bar.savefig('Data_balance_{}_prototypes.png'.format(hw_many))

	total_acc = np.zeros(shape=(3,2))
	for i, lvq in enumerate(lvqs):

		data = dataset
		for shu in range(10):
			np.random.shuffle(data)

		before = time.time()					
		prototypes = lvq(data, protos, hw_many, weight=False)
		ts = time.time() - before

		acc = np.arange(2)
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
			print((ok/len(pred)))
			total_acc[i][j] = (ok/len(pred))

	knns = plt.figure(2)
	x = np.arange(3)
	knn1, = plt.plot(total_acc[:,0], label="1-NN", linestyle='--')
	knn3, = plt.plot(total_acc[:,1], label="3-NN", linewidth=2)
	plt.legend(handles=[knn1, knn3])
	plt.xticks(x,('LVQ1', 'LVQ2.1', 'LVQ3'))
	plt.title('Accuracy comparrisson between LVQs')
	knns.savefig('Accuracy_comparrisson_{}_prototypes.png'.format(hw_many))