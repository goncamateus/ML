import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from source.pca import PCA


def load_dataset(db_name):
    db_path = os.path.join(
        os.path.abspath(os.path.pardir), 'databases')
    if(db_name.endswith('.arff')):
        db = arff.loadarff(os.path.join(db_path, db_name))
        df = pd.DataFrame(db[0])
    else:
        df = pd.read_csv(os.path.join(db_path, db_name))

    df_norm = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / \
        (df.iloc[:, :-1].max() - df.iloc[:, :-1].min())
    values = df.get_values()
    values[:, :-1] = df_norm

    for value in values:
        try:
            b = value[-1].decode('utf-8')
        except AttributeError:
            b = value[-1]

        if (b in ['false', 'no']):
            value[-1] = 0
        else:
            value[-1] = 1

    return values


if __name__ == '__main__':

    dataset = load_dataset('kc1.arff')
    data = dataset

    for shu in range(30):
        np.random.shuffle(data)

    X = data[:, :-1]
    Y = data[:, -1]

    # for i in range(1, np.shape(X)[1]+1):
    pca = PCA(X, k=5)
    pca_data = pca.run()

    X_train, X_test = pca_data[:int(len(pca_data)*0.7)], pca_data[int(len(pca_data)*0.7):]
    y_train, y_test = Y[:int(len(pca_data)*0.7)].astype(int), Y[int(len(pca_data)*0.7):].astype(int)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print("For " + str(20) +
          " dimensions, the accuracy was: " + str(accuracy_score(y_test, pred)))
