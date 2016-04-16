"""
TrainTestSets.py.

Holds the DataSet class.
"""
from attributes import Attributes
import os

from all_exceptions import LogicError

from example import Examples
from dataset import DataSet


class TrainTestSets(object):

    _test = None
    _train = None
    test_path = None
    train_path = None

    def __init__(self, test_path=None, train_path=None, args=None):
        """Initialize TrainTestSets with a testing filepath and training filepath."""
        if test_path or train_path:
            self.test_path = test_path
            self.train_path = train_path
            self.main()
        elif args:
            self.set_options(args)
            self.main()

    @property
    def test_set(self): # noqa
        return self._test

    @property
    def train(self):  # noqa
        return self._train

    def set_train(self, train):
        """Set train dataset."""
        self._train = train

    def set_test(self, test):
        """Set test dataset."""
        self._test = test

    def set_options(self, options):
        """Set options from arguments."""
        if type(options) is str:
            args = options.split(" ")
        i = 0
        if len(args) < 1:
            e = "LogicError: Incorrect number of arguments."
            raise Exception(e)
        if "-t" in args:
            while(args[i] != "-t"):
                if (i + 1) == len(args):
                    e = "LogicError: Incorrect number of arguments."
                    raise Exception(e)
                i += 1
            self.train_path = args[i + 1]
        if "-T" in args:
            while(args[i] != "-T"):
                if (i + 1) == len(args):
                    e = "LogicError: Incorrect number of arguments."
                    raise Exception(e)
                i += 1
            self.test_path = args[i + 1]

    def __str__(self):
        """String representation of TrainTestSet."""
        if self.test_set and self.train:
            return str(self.test_set) + "\n" + str(self.train)
        elif self.test_set:
            return str(self.test_set)
        elif self.train:
            return str(self.train)
        else:
            return "No data loaded."

    def main(self, args=None):
        """Main function for the testtrainset."""
        if not self.test_path and not self.train_path:
            if not args:
                raise LogicError("Set options first.")
            self.set_options(args)
        if self.test_path:
            test = self._create_data_set(self.test_path)
            self.set_test(test)
        if self.train_path:
            train = self._create_data_set(self.train_path)
            self.set_train(train)

    def _create_data_set(self, path):
        if not os.path.exists(path):
            raise Exception("LogicError: %s does not exist!" % path)
        with open(path) as f:
            attributes = Attributes()
            at_examples = False
            for line in f:
                line = line.replace("\n", "")
                if '@dataset' in line:
                    l = line.split(" ")
                    name = l[1]
                if '@attribute' in line:
                    attributes.parse(line)
                if '@example' in line:
                    at_examples = True
                    examples = Examples(attributes)
                    dataset = DataSet(name=name, attributes=attributes)
                    continue
                if at_examples and len(line) > 1:
                    examples.parse(line)
            dataset._examples = examples
            return dataset
