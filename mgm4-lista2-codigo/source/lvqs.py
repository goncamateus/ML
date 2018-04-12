import numpy as np
from sklearn import neighbors
from distances import euclidian_distance

def lvq1(dataset, protos, hw_many, weight=False):
	x = dataset[:,:-1]
	y = dataset[:, -1]
	prototypes = np.copy(protos)
	for k in range(hw_many):
		knn = neighbors.NearestNeighbors(n_neighbors=1)
		alfa = 0.1 * (1.0 - (k/float(hw_many)))
		for i in range(len(x)):
			knn.fit(prototypes[:, :-1])
			_, index = knn.kneighbors([x[i]])
			if prototypes[index[0][0]][-1] == y[i]:
				prototypes[index[0][0]][:-1] = prototypes[index[0][0]][:-1] + alfa*(x[i] - prototypes[index[0][0]][:-1])
			else:
				prototypes[index[0][0]][:-1] = prototypes[index[0][0]][:-1] - alfa*(x[i] - prototypes[index[0][0]][:-1])

	return prototypes

def in_window(test, prot1, prot2, w):
	
	di = euclidian_distance(test, prot1, test.size)
	dj = euclidian_distance(test, prot2, test.size)
	mini = min(di/dj, dj/di)
	s = ((1-w)/(1+w))

	return (mini > s)

def lvq21(dataset, protos, hw_many, weight=False):

	x = dataset[:,:-1]
	y = dataset[:, -1]
	
	prototypes = np.copy(protos)
	prot = [0,0]

	for k in range(hw_many):

		alfa = 0.1 * (1.0 - (k/float(hw_many)))
		knn = neighbors.NearestNeighbors(n_neighbors=2)

		for i in range(len(x)):

			knn.fit(prototypes[:, :-1], prototypes[:,-1])
			_, index = knn.kneighbors([x[i]])
			prot[0] = prototypes[index[0][0]]
			prot[1] = prototypes[index[0][1]]

			if in_window(x[i], prot[0], prot[1], w=0.2):

				if prot[0][-1] != prot[1][-1]:

					if prot[0][-1] == y[i]:
						prot[0][:-1] = prot[0][:-1] + alfa*(x[i] - prot[0][:-1])
						prot[1][:-1] = prot[1][:-1] - alfa*(x[i] - prot[1][:-1])

					elif prot[1][-1] == y[i]:
						prot[1][:-1] = prot[1][:-1] + alfa*(x[i] - prot[1][:-1])
						prot[0][:-1] = prot[0][:-1] - alfa*(x[i] - prot[0][:-1])

			prototypes[index[0][0]] = prot[0]
			prototypes[index[0][1]] = prot[1]

	return prototypes

def lvq3(dataset, protos, hw_many, weight=False):

	x = dataset[:,:-1]
	y = dataset[:, -1]
	
	prototypes = np.copy(protos)
	prot = [0,0]

	for k in range(hw_many):

		alfa = 0.1 * (1.0 - (k/float(hw_many)))
		knn = neighbors.NearestNeighbors(n_neighbors=2)

		for i in range(len(x)):

			knn.fit(prototypes[:, :-1], prototypes[:,-1])
			_, index = knn.kneighbors([x[i]])
			prot[0] = prototypes[index[0][0]]
			prot[1] = prototypes[index[0][1]]

			if in_window(x[i], prot[0], prot[1], w=0.2):

				if prot[0][-1] != prot[1][-1]:

					if prot[0][-1] == y[i]:
						prot[0][:-1] = prot[0][:-1] + alfa*(x[i] - prot[0][:-1])
						prot[1][:-1] = prot[1][:-1] - alfa*(x[i] - prot[1][:-1])

					elif prot[1][-1] == y[i]:
						prot[1][:-1] = prot[1][:-1] + alfa*(x[i] - prot[1][:-1])
						prot[0][:-1] = prot[0][:-1] - alfa*(x[i] - prot[0][:-1])

				elif prot[0][-1] == y[i]:
					prot[0][:-1] = prot[0][:-1] + 0.1*alfa*(prot[0][:-1] - x[i])
					prot[1][:-1] = prot[1][:-1] + 0.1*alfa*(prot[1][:-1] - x[i])

			prototypes[index[0][0]] = prot[0]
			prototypes[index[0][1]] = prot[1]

	return prototypes