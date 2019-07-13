import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class Imdb:
    def __init__(self):
        pass

    def create_model(self):
        train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

        (train_data, validation_data), test_data = tfds.load(
            name="imdb_reviews",
            split=(train_validation_split, tfds.Split.TEST),
            as_supervised=True)
        # 데이터 탐색

        train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
        # 10개의 샘플
        # print('10개의 샘플 %s'% (train_examples_batch))
        # print('10개의 라벨 %s' % (train_labels_batch))
        embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        hub_layer = hub.KerasLayer(embedding, input_shape=[],
                                   dtype=tf.string, trainable=True)
        hub_layer(train_examples_batch[:3])
        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        print(model.summary())
