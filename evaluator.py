"""Evaluator.py."""
import math
import random
import logging

from dataset import DataSet


logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def mean(vals):
    """Calculate the mean for a set of values."""
    return sum(vals) / float(len(vals))


def standard_deviation(vals):
    """Calculate stdev for a set of values."""
    average = mean(vals)
    stdev = sum((x - average)**2 for x in vals) / float(len(vals))
    return math.sqrt(stdev)


class Evaluator(object):
    """Class that checks the accruarcy."""

    def __init__(self, classifier, testing_set=None):
        """Init, k cross fold validation."""
        self._classifier = classifier
        if testing_set:
            self.testset = testing_set
            self.evaluate_testset()

    @property
    def classifier(self):
        """Return the classifier."""
        return self._classifier

    @property
    def trainset(self):
        """Training set of the classifier."""
        return self.classifier.trainset

    @trainset.setter
    def trainset(self, value):
        """Return testing set from self._set.train."""
        self._trainset = value

    @property
    def testset(self):
        """Test set of the classifier."""
        return self._testset

    @testset.setter
    def testset(self, value):
        """Return testing set from self._set.train."""
        self._testset = value

    @property
    def attributes(self):
        """Return attributes of the training dataset."""
        return self.trainset.attributes

    @attributes.setter
    def attributes(self, val):
        """Return attributes of the training dataset."""
        self.trainset.attributes = val

    @property
    def accuracy(self):
        """Return how many predictions were correct."""
        accurate = 0.0
        for example in self.testset.examples:
            label = self.classifier.classify(example=example)
            if example[-1] == label:
                accurate += 1.0
        accuracy = accurate / self.testset.examples_size
        return accuracy

    @property
    def split_size(self):
        """Split size for cross validation."""
        return math.ceil(self.trainset.examples_size / float(self.k))

    def _holdout_buckets(self, p):
        buckets = [[], []]
        test_size = int(math.ceil(self.trainset.examples_size * p))
        random.seed()
        sample = random.sample(range(0, self.trainset.examples_size), test_size)
        for i in xrange(self.trainset.examples_size):
            if i not in sample:
                buckets[0].append(self.trainset.examples[i])
            else:  # other dataset
                buckets[1].append(self.trainset.examples[i])
        return buckets

    def holdout(self, p):
        """Holdout evaluation."""
        self.classifier.add_datasets(self.trainset)
        folds = self._holdout_buckets(p)
        order = [[0, 1], [1, 0]]
        folds_errors = []
        folds_accuracy = []
        # create training dataset from all folds but folds[i]
        for i, k in order:
            trainset = DataSet(attributes=self.trainset.attributes, examples=folds[i])
            # classify using the classifier
            self.classifier.train(trainset)
            current_errors = 0.0
            for example in folds[k]:
                label = self.classifier.classify(example=example)
                if example[-1] != label:
                    current_errors += 1.0
                print(str(example), label)
            n = float(len(folds[i]))
            accuracy = (n - current_errors) / n
            folds_accuracy.append(accuracy)
            folds_errors.append(current_errors / n)
        mean_accuracy = mean(folds_accuracy)
        mean_errors = mean(folds_errors)
        print("%s\tp=%s" % (self.classifier.model, p))
        print("Average Accuracy: %s" % mean_accuracy)
        print("Average Error: %s" % mean_errors)
        print("Stdev: %s" % standard_deviation(folds_accuracy))

    def create_folds(self):
        """Return examples in folds."""
        folds = [[] for i in range(0, self.k)]
        for e in self.trainset.examples:
            # randomly assign to a fold until assigned to a fold that needs more examples
            rand = random.randint(0, self.k - 1)
            while(len(folds[rand]) >= self.split_size):
                rand = random.randint(0, self.k - 1)
            folds[rand].append(e)
        return folds

    def cross_validate(self, k=10):
        """Run the cross validation."""
        self.k = k
        folds_errors = []
        folds_accuracy = []
        folds = self.create_folds()
        # self.original_trainset = self.trainset
        for i in xrange(self.k):
            # create training dataset from all folds but folds[i]
            training = [folds[j] for j in xrange(self.k) if j != i]
            training = [example for sublist in training for example in sublist]  # merge into one list
            trainset = DataSet(attributes=self.trainset.attributes, examples=training)

            # classify using the classifier
            self.classifier.train(trainset)
            current_errors = 0.0
            for example in folds[i]:
                label = self.classifier.classify(example=example)
                if example[-1] != label:
                    current_errors += 1.0
                print(example, label)
            n = float(len(folds[i]))
            accuracy = (n - current_errors) / n
            folds_accuracy.append(accuracy)
            folds_errors.append(current_errors / n)
        mean_accuracy = mean(folds_accuracy)
        mean_errors = mean(folds_errors)
        print("%s\tfolds=%s" % (self.classifier.model, self.k))
        print("Average Accuracy: %s" % mean_accuracy)
        print("Average Error: %s" % mean_errors)
        print("Stdev: %s" % standard_deviation(folds_accuracy))

    def evaluate_testset(self):
        """Test with testset."""
        self.classifier.train(self.trainset)
        current_errors = 0.0
        for example in self.testset.examples:
            label = self.classifier.classify(example=example)
            if example[-1] != label:
                current_errors += 1.0
            n = float(len(self.testset.examples_size))
            accuracy = (n - current_errors) / n
        print("%s" % (self.classifier.model))
        print("Accuracy: %s" % accuracy)
