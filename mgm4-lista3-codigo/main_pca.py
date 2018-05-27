import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA as sklearnPCA
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
    total_acc = np.zeros(shape=(3, 2))
    data = dataset

    for shu in range(30):
        np.random.shuffle(data)

    X = data[:, :-1]
    Y = data[:, -1]

    # for i in range(1, np.shape(X)[1]+1):
    pca = PCA(X, k=10)
    pca_data = pca.run()
    sklearn_pca = sklearnPCA(n_components=10)
    Y_sklearn = new_x = sklearn_pca.fit_transform(X)
    print(pca_data)
    # print(Y_sklearn)
    # print('\nPara i =', i, ':\n', pca_data)

    # data_train = prototypes
    # data_test = data
    # y_test = data_test[:,-1]

    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(data_train[:, :-1], data_train[:,-1])
    # pred = knn.predict(data_test[:, :-1])
    # ok = 0
    # for t in range(len(pred)):
    #     if pred[t]  == y_test[t]:
    #         ok += 1
    # print((ok/len(pred)))
    # total_acc[i][j] = (ok/len(pred))

    # knns = plt.figure(2)
    # x = np.arange(3)
    # knn1, = plt.plot(total_acc[:,0], label="1-NN", linestyle='--')
    # knn3, = plt.plot(total_acc[:,1], label="3-NN", linewidth=2)
    # plt.legend(handles=[knn1, knn3])
    # plt.ylim(0.7, 0.9)
    # plt.xticks(x,('LVQ1', 'LVQ2.1', 'LVQ3'))
    # plt.title('Accuracy comparrisson between LVQs')
    # knns.savefig('Accuracy_comparrisson_{}_prototypes.png'.format(hw_many))
