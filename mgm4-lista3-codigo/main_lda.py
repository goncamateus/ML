import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from scipy.io import arff
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from source.lda import LDA


def load_dataset(db_name):
    db_path = os.path.join(
        os.path.abspath(os.path.pardir), 'databases')
    if(db_name.endswith('.arff')):
        db = arff.loadarff(os.path.join(db_path, db_name))
        df = pd.DataFrame(db[0])
    else:
        df = pd.read_csv(os.path.join(db_path, db_name))

    values = df.get_values()

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
    smote_enn = SMOTEENN(random_state=0)
    Y = data[:, -1]

    X_resampled, y_resampled = smote_enn.fit_sample(
        data[:, :-1].astype(float), Y.astype(int))
    data = np.concatenate(
        (X_resampled, y_resampled.reshape(y_resampled.shape[0], 1)), axis=1)
    Y = data[:, -1]

    for shu in range(30):
        np.random.shuffle(data)

    lda = LDA(data, k=1)
    lda_data = lda.run()

    X_train, X_test = lda_data[:int(
        len(lda_data)*0.7)], lda_data[int(len(lda_data)*0.7):]
    y_train, y_test = Y[:int(len(lda_data)*0.7)
                        ].astype(int), Y[int(len(lda_data)*0.7):].astype(int)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)  

    total_pred = [0,0]
    total_pred[1] = accuracy_score(y_test, pred)

    X_train, X_test = data[:int(
        len(data)*0.7)], data[int(len(data)*0.7):]
    y_train, y_test = Y[:int(len(data)*0.7)
                        ].astype(int), Y[int(len(data)*0.7):].astype(int)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test) 

    total_pred[0] = accuracy_score(y_test, pred)

    plt.bar([0,1],total_pred, 0.3)
    plt.xticks(np.array([x for x in range(2)]),
               ['Pure','LDA'])
    plt.ylabel('Accuracy')
    plt.savefig('KNN_{}_{}.png'.format('LDA', 'kc1'))

