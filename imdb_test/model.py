import tensorflow as tf
from tensorflow import keras
import numpy as np

class Imdb:
    def __init__(self):
        pass

    def create_model(self):
        imdb = keras.datasets.imdb
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))
        print("검증 샘플: {}, 레이블: {}".format(len(test_data), len(test_data)))
        # 첫번째 리뷰
        print(train_data[0])
