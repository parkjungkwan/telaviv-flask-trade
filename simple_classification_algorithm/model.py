import pandas as pd
import numpy as np
from simple_classification_algorithm.perceptron import Perceptron
class IrisModel:
    def __init__(self):
        self.iris = pd.read_csv('https://archive.ics.uci.edu/ml/'
                             'machine-learning-databases/iris/iris.data', header=None)
        # setosa 와 versicolor 를 선택
        t = self.iris.iloc[0:100, 4].values
        self.y = np.where(t == 'Iris-setosa', -1, 1)
        # 꽃받침 길이와 꽃임 길이를 추출
        self.X = self.iris.iloc[0:100, [0, 2]].values
        self.classfier_algorithm = Perceptron(eta = 0.1, n_iter= 10)

    def get_iris(self):
        return self.iris