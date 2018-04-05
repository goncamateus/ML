from knn import KNN
from prototype import generation
from distances import euclidian_distance

def lvq1(dataset, hw_many, weight=False):
	ds = dataset.tolist()
	prototypes = generation(dataset, hw_many)
	for k,_ in enumerate(hw_many):
		for x in ds:
			knn = KNN()
			neighbours = knn.get_neighbours(ds, prototypes[k], k=1, weight=weight)
			if knn.predict(neighbours) != x[-1]:
				#DUVIDA NA LINHA DE BAIXO - APROXIMO ALGUM PARAMETRO OU A CLASSE??
				prototypes[k][-1] = prototypes[k][-1] + 0.1*euclidian_distance(x, prototypes, x.size)
			else:
				prototypes[k][-1] = prototypes[k][-1] - 0.1*euclidian_distance(x, prototypes, x.size)
	return prototypes

def lvq21(dataset):
	return prototypes

def lvq3(dataset):
	return prototypes