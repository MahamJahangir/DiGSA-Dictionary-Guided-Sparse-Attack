import numpy as np
import tensorflow as tf
num_classes = 10

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, x_test, y_train, y_test):
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train /= 255
    x_test /= 255
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    return x_train, x_test, y_train, y_test
