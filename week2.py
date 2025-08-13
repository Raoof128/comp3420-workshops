import numpy as np
from matplotlib import pyplot as plt

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn as nn


# Exercises related to MNIST data set

def summary_mnist(train_labels, test_labels, labels_list):
    """Return a dictionary that counts the number of samples for each possible label
    >>> summary_mnist([0, 1, 2, 2, 1, 0, 0, 0, 2], [2, 0, 1, 1], range(3))
    {'train_labels_counts': [4, 2, 3], 'test_labels_counts': [1, 2, 1]}
    """

    return {}

def build_mnist_model(hidden_size, dropout_rate=0):
    """Return a Keras model that has 1 hidden layer based on these parameters:
      - hidden_size: Size of hidden layer
      - dropout_rate: Dropout rate of the dropout layer after the hidden layer.
          If the dropout rate is zero, then there should not be a dropout layer
    >>> model = build_mnist_model(100, 0.5)
    >>> len(model)
    4
    >>> isinstance(model[0], torch.nn.Linear)
    True
    >>> isinstance(model[1], torch.nn.ReLU)
    True
    >>> isinstance(model[2], torch.nn.Dropout)
    True
    >>> isinstance(model[3], torch.nn.Linear)
    True
    >>> model[0].in_features
    784
    >>> model[0].out_features
    100
    >>> model[3].out_features
    10
    >>> model[2].p
    0.5
"""
    return None

if __name__ == "__main__":
    import doctest
    doctest.testmod()