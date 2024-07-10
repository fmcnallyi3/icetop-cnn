import argparse

from matplotlib import pyplot as plt
import tensorflow as tf

plt.axis('off')

(training_data, training_labels), (testing_data, testing_labels) = tf.keras.datasets.mnist.load_data()

def show_selection(l):
    for i in l:
        plt.imshow(testing_data[i].reshape(28,28), cmap='gray')
        plt.show()

def show_range(start, end):
    for i in range(start, end):
        plt.imshow(testing_data[i].reshape(28,28), cmap='gray')
        plt.show()

show_range(100,110)