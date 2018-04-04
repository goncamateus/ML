import math

def euclidian_distance(in1, in2, hm_args):
	dist = 0
	for x in range(hm_args):
		dist += pow((in1[x] - in2[x]), 2)

	return math.sqrt(dist)

def vdm(test, hm_args, db_train, weight):

	Niac = [[dict() for _ in range(hm_args)] for i in range(2)]
	Nia = [dict() for _ in range(hm_args)]
	distances = list()

	for row in db_train:
		for i, var in enumerate(row[:-1]):
		
			if bool(Nia[i].get(var)):
				Nia[i][var] += 1
			else:
				Nia[i][var] = 1
			
			row[-1] = int(row[-1])
			if row[-1] == 0:
				if bool(Niac[0][i].get(var)):
					Niac[0][i][var] += 1
				else:
					Niac[0][i][var] = 1
			else:
				if bool(Niac[1][i].get(var)):
					Niac[1][i][var] += 1
				else:
					Niac[1][i][var] = 1

	for row in db_train:	
		total = 0
		for n in range(hm_args):
			small_vdm = 0
			for i in range(2):
				test_Niac = Niac[i][n].get(test[n])
				test_Nia = Nia[n].get(test[n])
				if (test_Nia == None or test_Nia == 0 or test_Niac == None or test_Niac == 0):
					piac = 0
				else:
					piac = test_Niac/test_Nia
				train_Niac = Niac[i][n].get(row[n])
				train_Nia = Nia[n].get(row[n])
				if (train_Nia == None or train_Nia == 0 or train_Niac == None or train_Niac == 0):
					pibc = 0
				else:
					pibc = train_Niac/train_Nia
				small_vdm += pow(piac - pibc, 2)
			total += small_vdm
		dist = math.sqrt(total)
		if weight:
			dist = dist * (1/(dist + 0.1))
		distances.append((row, dist))

	return distances
			
def hvdm(test, hm_args, db_train, weight):



	def preclass(db_train):
		categ = list()
		for i in db_train[0][:-1]:
			try:
				float(i)
				categ.append(1)
			except ValueError:
				categ.append(0)
		return categ

	def diff(x, y):
		return pow(x - y, 2)

	def in_vdm(test, index, db_train, weight, Niac, Nia, row):		
		small_vdm = 0
		for i in range(2):
			test_Niac = Niac[i][index].get(test[index])
			test_Nia = Nia[index].get(test[index])
			if (test_Nia == None or test_Nia == 0 or test_Niac == None or test_Niac == 0):
				piac = 0
			else:
				piac = test_Niac/test_Nia
			train_Niac = Niac[i][index].get(row[index])
			train_Nia = Nia[index].get(row[index])
			if (train_Nia == None or train_Nia == 0 or train_Niac == None or train_Niac == 0):
				pibc = 0
			else:
				pibc = train_Niac/train_Nia
			small_vdm += abs(piac - pibc)
		if weight:
			small_vdm = small_vdm * (1/(small_vdm + 0.1))
		return pow(small_vdm, 2)

		
	Niac = [[dict() for _ in range(hm_args)] for i in range(2)]
	Nia = [dict() for _ in range(hm_args)]
	distances = list()

	for row in db_train:
		for i, var in enumerate(row[:-1]):
		
			if bool(Nia[i].get(var)):
				Nia[i][var] += 1
			else:
				Nia[i][var] = 1
			
			row[-1] = int(row[-1])
			if row[-1] == 0:
				if bool(Niac[0][i].get(var)):
					Niac[0][i][var] += 1
				else:
					Niac[0][i][var] = 1
			else:
				if bool(Niac[1][i].get(var)):
					Niac[1][i][var] += 1
				else:
					Niac[1][i][var] = 1

	pc = preclass(db_train)
	
	for row in db_train:
		dist = 0
		for i, p in enumerate(pc):
			if p == 1:
				test[i] = float(test[i])
				row[i] = float(row[i])
				if weight:
					di = math.sqrt(diff(test[i], row[i]))
					dist += pow(di/(di + 0.1), 2)
				else:
					dist += diff(test[i], row[i])
			else:
				dist += in_vdm(test, i, db_train, weight, Niac, Nia, row)


		distances.append((row, math.sqrt(dist)))
	return distances
