import numpy as np
from random import randint, randrange

def generation(dataset, k, classes):

	prototypes = np.zeros(shape=(k,dataset[0].size))
	size = dataset[0].size
	balance = [0 for _ in range(classes[-1])]
	for i in range(k):
		for j in range(size):
			numb = dataset[randrange(len(dataset))][j]
			prototypes[i][j] = numb

		c1 = randint(1, classes[-1])
		prototypes[i][-1] = c1
		balance[c1-1] += 1
	return prototypes, balance
	