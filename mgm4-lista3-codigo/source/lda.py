import numpy as np
from sklearn.preprocessing import StandardScaler


class LDA:
    _mean_vecs = list()
    _no = list()
    _yes = list()
    _eigenvalues = _eigenvectors = None

    def __init__(self, data, k):
        # if normalized --> StandardScaler().fit_transform(data[:, :-1].astype(float))
        self._X = data[:, :-1].astype(float)
        self._Y = data[:, -1]
        self._k = k

    def calc_mean_vecs(self):
        self._no = [self._X[i] for i in range(len(self._Y)) if self._Y[i] == 0]
        self._yes = [self._X[i]
                     for i in range(len(self._Y)) if self._Y[i] == 1]
        self._mean_vecs.append(np.mean(self._no, axis=0))
        self._mean_vecs.append(np.mean(self._yes, axis=0))
        self._mean_vecs = np.array(self._mean_vecs)
        # print(self._mean_vecs)

    def calc_within_scatter_mat(self):
        shape = (np.shape(self._X)[1], 1)
        s_shape = (np.shape(self._X)[1], np.shape(self._X)[1])
        Sw = np.zeros(s_shape)
        Sc = np.zeros(s_shape)
        for x in self._no:
            mv = self._mean_vecs[0].reshape(shape)
            sub = x.reshape(shape) - mv
            Sc += sub.dot(sub.T)
        Sw += Sc
        Sc = np.zeros(s_shape)
        for x in self._yes:
            mv = self._mean_vecs[1].reshape(shape)
            sub = x.reshape(shape) - mv
            Sc += sub.dot(sub.T)
        Sw += Sc
        self._Sw = Sw

    def calc_between_scatter_mat(self):
        shape = (np.shape(self._X)[1], 1)
        average = np.mean(self._X, axis=0)
        average = average.reshape(shape)
        Sb = np.zeros((np.shape(self._X)[1], np.shape(self._X)[1]))
        Sb += len(self._no)*(self._mean_vecs[0].reshape(shape) -
                             average).dot((self._mean_vecs[0].reshape(shape) - average).T)
        Sb += len(self._yes) * \
            (self._mean_vecs[1].reshape(shape) -
             average).dot((self._mean_vecs[1].reshape(shape) - average).T)
        self._Sb = Sb

    def calc_eigenstuff(self):
        self._eigenvalues, self._eigenvectors = np.linalg.eig(np.linalg.inv(
            self._Sw).dot(self._Sb))

    def choose_best(self):
        eigenpairs = [(np.abs(self._eigenvalues[i]), self._eigenvectors[:, i])
                      for i in range(len(self._eigenvalues))]
        eigenpairs = sorted(eigenpairs, key=lambda k: k[0], reverse=True)
        return eigenpairs

    def run(self):
        self.calc_mean_vecs()
        self.calc_within_scatter_mat()
        self.calc_between_scatter_mat()
        self.calc_eigenstuff()
        pairs = self.choose_best()
        projection_matrix = [pairs[i][1] for i in range(self._k)]
        projection_matrix = np.array(projection_matrix).T
        final = self._X.dot(projection_matrix).real
        return final