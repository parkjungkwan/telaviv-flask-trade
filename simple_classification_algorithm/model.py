import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simple_classification_algorithm.perceptron import Perceptron
from matplotlib.colors import ListedColormap
class IrisModel:
    def __init__(self):
        self.iris = pd.read_csv('https://archive.ics.uci.edu/ml/'
                             'machine-learning-databases/iris/iris.data', header=None)
        # setosa 와 versicolor 를 선택
        t = self.iris.iloc[0:100, 4].values
        self.y = np.where(t == 'Iris-setosa', -1, 1)
        # 꽃받침 길이와 꽃잎를 추출
        self.X = self.iris.iloc[0:100, [0, 2]].values
        print('==================')
        print('X 값 %s' % self.X)
        self.classfier_algorithm = Perceptron(eta = 0.1, n_iter= 10)

    def get_iris(self):
        return self.iris

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def draw_scatter(self):
        X = self.X
        plt.scatter(X[:50,0], X[:50,1],
                    color = 'red',
                    marker='o',
                    label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='blue',
                    marker='x',
                    label='versicolor')
        plt.xlabel('sepal lenght[cm]')
        plt.ylabel('petal lenght[cm]')
        plt.legend(loc = 'upper left')
        plt.show()

    def draw_errors(self):
        X = self.X
        y = self.y
        self.classfier_algorithm.fit(X, y)
        plt.plot(range(1, len(self.classfier_algorithm.errors_) +1),
                 self.classfier_algorithm.errors_, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Number of errors')
        plt.show()

    def plot_decision_regions(self):
        X = self.X
        y = self.y
        classifier = Perceptron(eta=0.1, n_iter=10)
        classifier.fit(X, y)
        colors = ('red','blue', 'lightgreen','gray','cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # 결정 경계 그리기
        print('X 의 size : %d' % len(X))
        x1_min, x1_max = X[:,0].min() -1, X[:,0].max() +1
        x2_min, x2_max = X[:,1].min() -1, X[:,1].max() +1
        resolution = 0.2
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap = cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # 샘플의 산점도
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x = X[y == cl, 0],
                        y = X[y == cl, 1],
                        alpha=0.8,
                        c = colors[idx],
                        label = cl,
                        edgecolors= 'black')
        plt.xlabel('sepal length[cm]')
        plt.ylabel('petal length[cm]')
        plt.legend(loc = 'upper left')
        plt.show()

