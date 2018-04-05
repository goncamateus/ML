import numpy as np
from random import randint, randrange

def generation(dataset, k):

	args = np.zeros(shape=(dataset[0].size, dataset.size))
	for j, x in enumerate(dataset):
		for i, v in enumerate(x):
			args[i][j] = v

	maxes = np.zeros(shape=(dataset[0].size, 1))
	meds = np.zeros(shape=(dataset[0].size, 1))

	for i,_ in enumerate(args):
		maxes[i] = args[i].max()
		meds[i] = (maxes[i] - args[i].min())/2

	prototypes = np.zeros(shape=(k,dataset[0].size))

	for i in range(k):
		for j,_ in enumerate(maxes[:-1]):
			c1 = randint(1, 2)
			if c1 == 1:
				prototypes[i][j] = float(randint(0, int(meds[j])))
			else:
				prototypes[i][j] = float(randrange(int(meds[j]), int(maxes[j])))
		prototypes[i][-1] = randint(0, 1)

	return prototypes
	