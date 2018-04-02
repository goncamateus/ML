import math


def euclidian_distance(in1, in2, hm_args):
	dist = 0
	for x in range(hm_args):
		dist += pow((in1[x] - in2[x]), 2)

	return math.sqrt(dist)
