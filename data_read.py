from scipy.io import arff
import pandas as pd
import os

db_path = os.path.join(os.path.abspath(os.path.curdir), 'databases')

datatrieve = arff.loadarff(os.path.join(db_path, os.listdir(db_path)[0]))
datatriveframe = pd.DataFrame(datatrieve[0])

desharnais = arff.loadarff(os.path.join(db_path, os.listdir(db_path)[2]))
desharnaisframe = pd.DataFrame(desharnais[0])

