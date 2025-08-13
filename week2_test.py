import numpy as np
import unittest

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from torchvision import datasets

import week2

class TestMNIST(unittest.TestCase):
    def test_summary_mnist(self):
        # Test the function with the MNIST dataset
        data_folder = "MNIST"
        train_mnist = datasets.MNIST(data_folder, download=True, train=True)
        test_mnist = datasets.MNIST(data_folder, download=True, train=False)

        train_targets = train_mnist.targets
        test_targets = test_mnist.targets
        result = week2.summary_mnist(train_targets.tolist(), test_targets.tolist(), range(10))
        self.assertDictEqual(result, 
                             {'train_labels_counts': [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], 
                              'test_labels_counts': [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]})
 
    def test_build_mnist_model(self):
        # Test the model
        model = week2.build_mnist_model(100, 0.5)
        self.assertEqual(len(model), 4)
        self.assertIsInstance(model[0], torch.nn.Linear)
        self.assertIsInstance(model[1], torch.nn.ReLU)
        self.assertIsInstance(model[2], torch.nn.Dropout)
        self.assertIsInstance(model[3], torch.nn.Linear)
        self.assertEqual(model[0].in_features, 28 * 28)
        self.assertEqual(model[0].out_features, 100)
        self.assertEqual(model[3].out_features, 10)
        self.assertEqual(model[2].p, 0.5)

        model = week2.build_mnist_model(40, 0)
        self.assertEqual(len(model), 3)
        self.assertIsInstance(model[0], torch.nn.Linear)
        self.assertIsInstance(model[1], torch.nn.ReLU)
        self.assertIsInstance(model[2], torch.nn.Linear)
        self.assertEqual(model[0].in_features, 28 * 28)
        self.assertEqual(model[0].out_features, 40)
        self.assertEqual(model[2].out_features, 10)


if __name__ == "__main__":
    unittest.main()