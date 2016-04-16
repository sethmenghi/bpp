"""
bayes.py.

Implementatino of NaiveBayes classifier object.
"""
import math
from collections import defaultdict

from classifier import Classifier


def mean(vals):
    """Calculate the mean for a set of values."""
    return sum(vals) / float(len(vals))


def standard_deviation(vals):
    """Calculate stdev for a set of values."""
    average = mean(vals)
    stdev = sum(float(float(x) - average)**2 for x in vals) / float(len(vals))
    return math.sqrt(stdev)


class NaiveBayes(Classifier):
    """NaiveBayes classifier."""

    def __init__(self, trainset, testset=None):
        """Initialize with dataset."""
        self.trainset = trainset
        self.testset = testset
        self.total_examples = self.train_set.examples.size

    def count_values(self):
        """Count each value for all training sets."""
        counts = defaultdict(lambda: defaultdict(list))
        for example in self.train_set.examples:  # for each example in the training set
            for i, v in enumerate(example.values):  # in each value
                print("Example: (%s), i: (%s), v: (%s)" % (str(example), i, v))
                counts[self.attributes[i].name][v].append(example)
        self.counts = counts
        return counts

    def predict(self, example, target_attribute):
        """Predict an example based on, predict target attribute."""
        self.counts[target_attribute]

    def train(self):
        """Train datasets."""
        classes = self.count_values()
        info = self.predict(classes)
