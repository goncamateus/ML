from source.knns import KNN
from source.distances import euclidian_distance
from sklearn.model_selection import KFold


def cross_validation(knn, hm_neighbours):
	ac = []
	kf = KFold(n_splits=10, shuffle=True)
	for k in hm_neighbours: 
		localac = []
		for train_index, test_index in kf.split(knn.folds):
			db_train = knn.folds[train_index]
			db_train = db_train.tolist()
			db_test = knn.folds[test_index]
			db_test = db_test.tolist()
			localac.append(0)				
			for test in db_test:
				neighbours = knn.get_neighbours(
					db_train, test, k, euclidian_distance, True)
				result = knn.predict(neighbours)
				if (int(test[-1])==result):
					localac[-1] += 1
			localac[-1] = localac[-1]/len(db_test)
		ac.append(0)
		for lac in localac:
			ac[-1] += lac
		ac[-1] /= len(localac)
		print('Accuracy for k={} :'.format(k), ac[-1])

	return ac

if __name__ == '__main__':	
	knn = KNN('kc2.arff')
	cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15])
