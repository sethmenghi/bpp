"""
classifier.py.

Holds the Classifier base class & Evaluator object.
"""
import logging

from all_exceptions import LogicError


logger = logging.getLogger(__name__)


class Classifier(object):
    """Base class for classifiers."""

    def __init__(self, trainset=None):
        """Training set for a classifier."""
        self.trainset = trainset

    def add_datasets(self, trainset=None, testset=None):
        """Set traintestset as self._set."""
        self._trainset = trainset
        self._testset = testset

    @property
    def trainset(self):
        """Return the TrainTestSet object in the Classifier."""
        return self._trainset

    @trainset.setter
    def trainset(self, value):
        """Return testing set from self._set.train."""
        self._trainset = value

    @property
    def testset(self):
        """Return the TrainTestSet object in the Classifier."""
        return self._testset

    @testset.setter
    def testset(self, value):
        """Return testing set from self._set.train."""
        self._testset = value

    @property
    def train_examples(self):
        """Return training examples."""
        return self.trainset.examples

    @property
    def attributes(self):
        """Return attributes."""
        return self.train_examples.attributes

    def train(self, dataset=None):
        """Train a dataset."""
        if dataset:
            self.trainset = dataset
        self._train()

    def _train(self):
        """Implemented in classifier objects."""
        pass

    def classify(self, example=None, dataset=None):
        """Classify a dataset or an example."""
        if example:  # classify an entire dataset
            class_label = self._classify(example=example)
        elif example:
            class_label = [self._classify(example=e) for e in dataset.examples]
        else:
            e = 'No example or dataset supplied for Classifier.classify'
            raise LogicError(e)
        return class_label
