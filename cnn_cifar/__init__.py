import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(X_train, y_train),(X_test, y_test) = keras.datasets.cifar10.load_data()
print(X_train.shape, y_train.shape)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

X_train = X_train / 255.0
X_test = X_test / 255.0

plt.imshow(X_train[0])
plt.imshow(X_test[0])