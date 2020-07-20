import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from source.lvqs import lvq1, lvq3, lvq21
from source.prototype import generation


def load_dataset(db_name):
    db_path = os.path.join(
        os.path.abspath(os.path.pardir), 'databases')
    if(db_name.endswith('.arff')):
        db = arff.loadarff(os.path.join(db_path, db_name))
        df = pd.DataFrame(db[0])
    else:
        df = pd.read_csv(os.path.join(db_path, db_name))
    sex = df['Sex']
    num_sex = list()
    for s in sex:
        if s == 'F':
            ns = 0
        elif s == 'M':
            ns = 1
        else:
            ns = 2
        num_sex.append(ns)
    df['Sex'] = np.array(num_sex)
    return df


if __name__ == '__main__':

    df = load_dataset('abalone.csv')
    classes = list(set(df['Rings'].tolist()))
    dataset = df.values

    lvqs = [lvq3]
    hw_many = 1000
    print('Selecting prototypes')
    protos, proto_balance = generation(dataset, hw_many, classes)

    x = dataset[:, :-1]
    y = dataset[:, -1]
    acc = list()
    for _ in range(30):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
    acc = np.mean(acc)
    print(acc)
    total_acc = np.zeros(shape=(3, 2))
    for i, lvq in enumerate(lvqs):

        data = dataset
        for shu in range(10):
            np.random.shuffle(data)

        before = time.time()
        print('LVQ')
        prototypes = lvq(data, protos, hw_many, weight=False)
        ts = time.time() - before
        print(f'Took {ts} s')

        # acc = np.arange(2)
        for j, k in enumerate([1]):

            for shu in range(10):
                np.random.shuffle(data)

            data_train = prototypes
            data_test = data
            y_test = data_test[:, -1]

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(data_train[:, :-1], data_train[:, -1])
            pred = knn.predict(data_test[:, :-1])
            report = accuracy_score(y_test, pred)
            print(f'{k}-NN with LVQ:\n', report)

