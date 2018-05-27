import numpy as np
from sklearn.preprocessing import StandardScaler


class PCA():
    _eigenvalues = None
    _eigenvectors = None
    _cov_mat = None

    def __init__(self, data, k):
        self._data = StandardScaler().fit_transform(data.astype(float))
        self._k = k

    def calc_cov_matrix(self):
        self._cov_mat = np.cov(self._data.T)

    def get_cov_matrix(self):
        return self._cov_mat

    def calc_eigenstuff(self):
        self._eigenvalues, self._eigenvectors = np.linalg.eig(self._cov_mat)

    def get_eigenvalues(self):
        return self._eigenvalues

    def get_eigenvectors(self):
        return self._eigenvectors

    def choose_best(self):
        eigenpairs = [(np.abs(self.get_eigenvalues()[i]), self.get_eigenvectors()[:, i])
                      for i in range(len(self.get_eigenvalues()))]
        eigenpairs.sort()
        eigenpairs.reverse()
        return eigenpairs

    def run(self):
        self.calc_cov_matrix()
        self.calc_eigenstuff()
        pairs = self.choose_best()
        projection_matrix = [pairs[i][1] for i in range(self._k)]
        projection_matrix = np.array(projection_matrix).T
        return projection_matrix
