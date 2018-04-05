import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from scipy.io import arff
from source.lvqs import lvq1, lvq21, lvq3
from source.prototype import generation
from sklearn.neighbors import KNeighborsClassifier as KNN



def load_dataset(db_name):
	db_path = os.path.join(
			os.path.abspath(os.path.pardir), 'databases')
	if(self.dataset.endswith('.arff')):
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

	return dataset

def accuracy_score(y_test, pred):
	accuracy = 0
	for i, acc in enumerate(y_test):
		if acc == pred[i]:
			accuracy += 1

	accuracy /= len(y_test)
	return accuracy

if __name__ == '__main__':
	
	dataset = load_dataset('kc1.arff')

	lvqs = [lvq1, lvq21, lvq3]
	scores = np.array([])

	fig, ax = plt.subplots()
	image, = ax.plot()
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.draw()
	
	for i, lvq in enumerate(lvqs):
		
		before = time.time()		
		data_train = lvq(dataset)
		ts = time.time() - before

		X = data_train[:, :-1]		
		y = data_train[:, -1]

		X_train, X_test, y_train, y_test = \
			train_test_split(X, y, test_size=0.33, random_state=42)

		knn = KNN(n_neighbors=1)
		knn.fit(X_train, y_train)

		pred = knn.predict(X_test)
		scores[i] = np.append(scores[i], accuracy_score(y_test, pred))
		print(scores[i])

		image.set_data(scores[i])
		plt.suptitle(lvq.__name__, fontsize=14, fontweight='bold')
		plt.title('Timestamp: {} seconds'.format(ts))
		plt.draw()
		fig.savefig("{}.png".format(lvq.__name__))
