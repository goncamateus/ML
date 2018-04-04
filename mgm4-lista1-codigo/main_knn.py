import matplotlib.pyplot as plt
import numpy as np
import time
from source.knns import KNN
from source.distances import euclidian_distance, vdm, hvdm
from sklearn.model_selection import KFold


def cross_validation(knn, hm_neighbours, weight, distance):
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
			for i, test in enumerate(db_test):
				#print('Test:\n',test)
				neighbours = knn.get_neighbours(
					db_train, test, k, distance, weight)
				#print('Neighbours:\n', neighbours)
				result = knn.predict(neighbours)
				#print(test[-1], result)
				test[-1] = float(test[-1])
				if (test[-1]==result):
					local_accuracy[-1] += 1
					#print(local_accuracy)
			local_accuracy[-1] = local_accuracy[-1]/len(db_test)
		accuracy.append(0)
		for lac in local_accuracy:
			accuracy[-1] += lac
		accuracy[-1] /= len(local_accuracy)
		print('Accuracy for k={} :'.format(k), accuracy[-1])

	return accuracy

if __name__ == '__main__':

	tot = time.time()
#--------------------------------------------------First dataset---------------------------------------------------
	print('Dataset {}\n\n'.format('kc1.arff'))
	knn = KNN('kc1.arff')
	print('{} Unweighted'.format(euclidian_distance.__name__))
	prev = time.time()
	cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight=False, distance=euclidian_distance)
	ts = int(time.time() - prev)
	plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	plt.plot(cv)
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.suptitle('Unweighted', fontsize=14, fontweight='bold')
	plt.title('Timestamp: {} seconds'.format(ts))
	plt.savefig('KNN_{}_Unweighted_{}.png'.format(euclidian_distance.__name__, 'kc1.arff'.replace('.arff','')))
	
	print('Weighted')
	prev = time.time()
	cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight= True, distance=euclidian_distance)
	ts = int(time.time() - prev)
	plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	plt.plot(cv)
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.suptitle('Weighted', fontsize=14, fontweight='bold')
	plt.title('Timestamp: {} seconds'.format(ts))
	plt.savefig('KNN_{}_Weighted_{}.png'.format(euclidian_distance.__name__, 'kc1.arff'.replace('.arff','')))

#---------------------------------------------Second dataset--------------------------------------------------------------
	print('Dataset {}\n\n'.format('kc2.arff'))
	knn = KNN('kc2.arff')
	print('{} Unweighted'.format(euclidian_distance.__name__))
	prev = time.time()
	cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight=False, distance=euclidian_distance)
	ts = int(time.time() - prev)
	plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	plt.plot(cv)
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.suptitle('Unweighted', fontsize=14, fontweight='bold')
	plt.title('Timestamp: {} seconds'.format(ts))
	plt.savefig('KNN_{}_Unweighted_{}.png'.format(euclidian_distance.__name__, 'kc2.arff'.replace('.arff','')))
	
	print('Weighted')
	prev = time.time()
	cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight= True, distance=euclidian_distance)
	ts = int(time.time() - prev)
	plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	plt.plot(cv)
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.suptitle('Weighted', fontsize=14, fontweight='bold')
	plt.title('Timestamp: {} seconds'.format(ts))
	plt.savefig('KNN_{}_Weighted_{}.png'.format(euclidian_distance.__name__, 'kc2.arff'.replace('.arff','')))

#------------------------------------------------Third dataset----------------------------------------------------------------------------
	print('Dataset {}\n\n'.format('tictactoe'))
	knn = KNN('tictactoe.csv')
	print('{} Unweighted'.format(vdm.__name__))
	prev = time.time()
	cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight=False, distance=vdm)
	ts = int(time.time() - prev)
	plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	plt.plot(cv)
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.suptitle('Unweighted', fontsize=14, fontweight='bold')
	plt.title('Timestamp: {} seconds'.format(ts))
	plt.savefig('KNN_{}_Unweighted_{}.png'.format(vdm.__name__, 'tictactoe'.replace('.csv','')))

	print('{} Weighted'.format(vdm.__name__))
	prev = time.time()
	cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight=True, distance=vdm)
	ts = int(time.time() - prev)
	plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	plt.plot(cv)
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.suptitle('Weighted', fontsize=14, fontweight='bold')
	plt.title('Timestamp: {} seconds'.format(ts))
	plt.savefig('KNN_{}_weighted_{}.png'.format(vdm.__name__, 'tictactoe'.replace('.csv','')))

#--------------------------------------------Fourth dataset------------------------------------------------------------------------------
	print('Dataset {}\n\n'.format('balloons'))
	knn = KNN('balloons.csv')
	print('{} Unweighted'.format(vdm.__name__))
	prev = time.time()
	cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight=False, distance=vdm)
	ts = int(time.time() - prev)
	plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	plt.plot(cv)
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.suptitle('Unweighted', fontsize=14, fontweight='bold')
	plt.title('Timestamp: {} seconds'.format(ts))
	plt.savefig('KNN_{}_Unweighted_{}.png'.format(vdm.__name__, 'balloons'.replace('.csv','')))

	print('{} Weighted'.format(vdm.__name__))
	prev = time.time()
	cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight=True, distance=vdm)
	ts = int(time.time() - prev)
	plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	plt.plot(cv)
	plt.ylabel('Accuracy')
	plt.xlabel('Neighbours')
	plt.suptitle('Weighted', fontsize=14, fontweight='bold')
	plt.title('Timestamp: {} seconds'.format(ts))
	plt.savefig('KNN_{}_weighted_{}.png'.format(vdm.__name__, 'balloons'.replace('.csv','')))

#-----------------------------------------------Fifth dataset-------------------------------------------------------------------------
#-----------------------------------------------NOT WORKING---------------------------------------------------------------------------
	
	# print('Dataset {}\n\n'.format('crx'))
	# knn = KNN('crx.csv')
	# print('{} unweighted'.format(hvdm.__name__))
	# prev = time.time()
	# cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight=False, distance=hvdm)
	# ts = int(time.time() - prev)
	# plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	# plt.plot(cv)
	# plt.ylabel('Accuracy')
	# plt.xlabel('Neighbours')
	# plt.suptitle('unweighted', fontsize=14, fontweight='bold')
	# plt.title('Timestamp: {} seconds'.format(ts))
	# plt.savefig('KNN_{}_unweighted_{}.png'.format(hvdm.__name__, 'crx'.replace('.csv','')))

	# print('Dataset {}\n\n'.format('crx'))
	# knn = KNN('crx.csv')
	# print('{} Weighted'.format(hvdm.__name__))
	# prev = time.time()
	# cv = cross_validation(knn, hm_neighbours=[1,2,3,5,7,9,11,13,15], weight=True, distance=hvdm)
	# ts = int(time.time() - prev)
	# plt.xticks(np.array([x for x in range(15)]),[1,2,3,5,7,9,11,13,15])	
	# plt.plot(cv)
	# plt.ylabel('Accuracy')
	# plt.xlabel('Neighbours')
	# plt.suptitle('Weighted', fontsize=14, fontweight='bold')
	# plt.title('Timestamp: {} seconds'.format(ts))
	# plt.savefig('KNN_{}_Weighted_{}.png'.format(hvdm.__name__, 'crx'.replace('.csv','')))


	total = time.time() - tot
	print('Total time:', total)