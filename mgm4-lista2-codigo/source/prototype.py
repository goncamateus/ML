import numpy as np
from random import randint, randrange

def generation(dataset, k):

	prototypes = np.zeros(shape=(k,dataset[0].size))
	args = dataset[0].size
	for i in range(k):
		for j in range(args - 1):
			numb = dataset[randrange(len(dataset))][j]
			while(type(numb) != type(0.1)):
				numb = dataset[randrange(len(dataset))][j]
			prototypes[i][j] = numb

		prototypes[i][-1] = dataset[randrange(len(dataset))][-1]

	return prototypes
	