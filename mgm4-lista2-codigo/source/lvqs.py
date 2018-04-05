import numpy as np
from knn import KNN
from prototype import generation
from distances import euclidian_distance

def lvq1(dataset, hw_many, weight=False):
	ds = dataset
	prototypes = generation(dataset, hw_many)
	for k in range(hw_many):
		for _, x in enumerate(ds):
			knn = KNN()
			neighbours = knn.get_neighbours(prototypes, x, k=1, weight=weight)
			if knn.predict(neighbours) != x[-1]:
				for v in prototypes[k][:-1]:
					v = v + 0.1*euclidian_distance(x, prototypes[k], x.size)
			else:
				for v in prototypes[k][:-1]:
					v = v - 0.1*euclidian_distance(x, prototypes[k], x.size)
	return prototypes

def in_window(test, prot1, prot2, w):
	
	di = euclidian_distance(test, prot1, test.size)
	dj = euclidian_distance(test, prot2, test.size)
	mini = min(di/dj, dj/di)
	s = ((1-w)/(1+w))

	return (mini > s)

def lvq21(dataset, hw_many, weight=False):
	
	prototypes = lvq1(dataset, hw_many, weight=weight)
	prot = np.zeros(shape=(hw_many,dataset[0].size))
	for k in range(hw_many-1):
		for _, x in enumerate(dataset):
			knn = KNN()
			neighbours = knn.get_neighbours(prototypes, x, k=2, weight=weight)
			prot[k] = neighbours[0]
			prot[k+1] = neighbours[1]
			if in_window(x, prot[k], prot[k+1], w=0.2):
				if prot[k][-1] != prot[k+1][-1]:
					if prot[k][-1] == x[-1]:
						for v in prot[k][:-1]:
							v = v + 0.1*euclidian_distance(x, prot[k], x.size)
						for v in prot[k+1][:-1]:
							v = v - 0.1*euclidian_distance(x, prot[k+1], x.size)
					elif prot[k+1][-1] == x[-1]:
						for v in prot[k+1][:-1]:
							v = v + 0.1*euclidian_distance(x, prot[k+1], x.size)
						for v in prot[k][:-1]:
							v = v - 0.1*euclidian_distance(x, prot[k], x.size)
	return prot

def lvq3(dataset, hw_many, weight=False):

	prototypes = lvq1(dataset, hw_many, weight=weight)
	prot = np.zeros(shape=(hw_many,dataset[0].size))
	for k in range(hw_many-1):
		for _, x in enumerate(dataset):
			knn = KNN()
			neighbours = knn.get_neighbours(prototypes, x, k=2, weight=weight)
			prot[k] = neighbours[0]
			prot[k+1] = neighbours[1]
			if in_window(x, prot[k], prot[k+1], w=0.2):
				if prot[k][-1] != prot[k+1][-1]:
					if prot[k][-1] == x[-1]:
						for v in prot[k][:-1]:
							v = v + 0.1*euclidian_distance(x, prot[k], x.size)
						for v in prot[k+1][:-1]:
							v = v - 0.1*euclidian_distance(x, prot[k+1], x.size)
					elif prot[k+1][-1] == x[-1]:
						for v in prot[k+1][:-1]:
							v = v + 0.1*euclidian_distance(x, prot[k+1], x.size)
						for v in prot[k][:-1]:
							v = v - 0.1*euclidian_distance(x, prot[k], x.size)

				elif prot[k][-1] == x[-1]:
					for v in prot[k][:-1]:
						v = v + 0.1*0.1*euclidian_distance(x, prot[k], x.size)
					for v in prot[k+1][:-1]:
						v = v + 0.1*0.1*euclidian_distance(x, prot[k+1], x.size)
	return prot