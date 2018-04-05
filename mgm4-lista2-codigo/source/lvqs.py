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

def lvq21(dataset, hw_many, weight=False):
	
	def in_window(test, prot1, prot2, w):
		di = euclidian_distance(test, prot1)
		dj = euclidian_distance(test, prot2)
		mini = min(di/dj, dj/di)
		s = ((1-w)/(1+w))
		return (mini > s)

	prototypes = lvq1(dataset, hw_many, weight=weight)
	# for k in range(hw_many):
	# 	for _, x in dataset:

	return prototypes

def lvq3(dataset):
	prototypes = lvq1(dataset, hw_many, weight=weight)
	return prototypes