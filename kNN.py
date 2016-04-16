"""
kNN.py.

Implementation of k-nearest neighbor in O(k) space.
"""
import logging

from math import sqrt
from operator import itemgetter
from collections import defaultdict

from classifier import Classifier
from attributes import NominalAttribute


logger = logging.getLogger(__name__)


class NearestNeighbor(Classifier):  # noqa

    """k nearest neighbor class."""

    model = 'kNN'

    def __init__(self, trainset, k=3):
        """Initialize the dataset and k."""
        self._trainset = trainset
        self.k = k

    def _classify(self, example):
        """Find neighbor and predict based off training on an example."""
        neighbors = self.find_neighbor(example)
        class_label = self.find_response(neighbors)
        return class_label

    def find_neighbor(self, test_example):
        """Find `test_example's` closest neighbor."""
        n = test_example.size
        # Create list of tuples --> [(training_example, distance to test_example), ... ]
        dists = []
        for i, e in enumerate(self.train_examples):
            dist = self.euclidean_dist(test_example, e, n)
            class_value = self.train_examples.get_class_value_at(i)
            dists.append((class_value, dist))
        dists.sort(key=itemgetter(1))
        neighbors = [dists[i][0] for i in xrange(self.k)]  # closest self.k neighbors
        return neighbors

    def euclidean_dist(self, example1, example2, length):
        """Sum of squared differences between two examples."""
        dist = 0
        for i in xrange(example1.size - 1):
            nominal = isinstance(self.attributes[i], NominalAttribute)
            if not nominal:
                dist += (example1[i] - example2[i])**2
            elif nominal and (example2[i] != example1[i]):
                dist += 1
        return sqrt(dist)

    def find_response(self, neighbors):
        """Take a list of neighbors and classify from the neighbors of the kNN."""
        counts = defaultdict(int)
        for neighbor in neighbors:
            counts[neighbor] += 1  # last item in list is the predicted value
        response = max(counts.items())[0]
        return response
