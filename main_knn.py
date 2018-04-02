import matplotlib.pyplot as plt
import numpy as np
import time
from source.knns import KNN
from source.distances import euclidian_distance
from sklearn.model_selection import KFold


def cross_validation(knn, hm_neighbours, weight):
	accuracy = []
	kf = KFold(n_splits=10, shuffle=True)
	for k in hm_neighbours: 
		local_accuracy = []
		for train_index, test_index in kf.split(knn.folds):
			db_train = knn.folds[train_index]
			db_train = db_train.tolist()
			db_test = knn.folds[test_index]
			db_test = db_test.tolist()
			local_accuracy.append(0)				
			for test in db_test:
				neighbours = knn.get_neighbours(
					db_train, test, k, euclidian_distance, weight)
				result = knn.predict(neighbours)
				if (int(test[-1])==result):
					local_accuracy[-1] += 1
			local_accuracy[-1] = local_accuracy[-1]/len(db_test)
		accuracy.append(0)
		for lac in local_accuracy:
			accuracy[-1] += lac
		accuracy[-1] /= len(local_accuracy)
		print('Accuracy for k={} :'.format(k), accuracy[-1])

	return accuracy

if __name__ == '__main__':
	datasets = ['kc1.arff','kc2.arff']	
	for ds in datasets:
		print('Dataset {}\n\n'.format(ds))
		knn = KNN(ds)
		print('Unweighted')
		prev = time.time()
		cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight= False)
		ts = int(time.time() - prev)
		plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
		plt.plot(cv)
		plt.ylabel('Accuracy')
		plt.xlabel('Neighbours')
		plt.suptitle('Unweighted', fontsize=14, fontweight='bold')
		plt.title('Timestamp: {} seconds'.format(ts))
		plt.savefig('KNN_Unweighted_{}.png'.format(ds.replace('.arff','')))
		
		print('Weighted')
		prev = time.time()
		cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight= True)
		ts = int(time.time() - prev)
		plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
		plt.plot(cv)
		plt.ylabel('Accuracy')
		plt.xlabel('Neighbours')
		plt.suptitle('Weighted', fontsize=14, fontweight='bold')
		plt.title('Timestamp: {} seconds'.format(ts))
		plt.savefig('KNN_Weighted_{}.png'.format(ds.replace('.arff','')))
