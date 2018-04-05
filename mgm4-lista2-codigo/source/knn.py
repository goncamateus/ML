import math
from distances import euclidian_distance


class KNN:

	def get_neighbours(self, db_train, test, k, weight=False):
	distances = []
	args = len(test) - 1

	for x in db_train:
		dist = euclidian_distance(test, x, args)
		w = 1/(dist + 0.1)
		if weight:
			distances.append((x, dist*w))
		else:
			distances.append((x, dist))

	for x in range(k):
		neighbours.append(distances[x][0])
	return neighbours

	def predict(self, neighbours):
	class1 = 0
	class2 = 0

	for n in neighbours:
		if n[-1] == 0:
			class1 += 1
		else:
			class2 += 1
	if class1 > class2:
		return 0

	else:
		return 1
